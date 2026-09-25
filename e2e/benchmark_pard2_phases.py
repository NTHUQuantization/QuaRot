"""Adapt the existing generation algorithm to the teammate's phase harness.

For batch one, compile a local generator with one prefill boundary and no
internal instrumentation. Batched PARD2 uses the runtime's phase generator.
The shared benchmark owns timing and memory counters in either case.
"""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

import torch

from e2e import speculative
from e2e import benchmark_real_llama_runtime as harness


class UntimedStages:
    def record(self, name, fn):
        return fn()

    def totals(self):
        return {}


def phased_generator(method):
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    fn = tree.body[0]
    fn.decorator_list = []
    is_ar = fn.name == '_generate_ar'
    if not is_ar and fn.name != '_generate_spec':
        raise ValueError('Unsupported generation method')
    loops = [n for n in fn.body if isinstance(n, ast.While)]
    if len(loops) != 1:
        raise RuntimeError('Generation loop changed; review phase split')
    loop = loops[0]
    boundary = ast.Expr(value=ast.Yield(value=ast.Constant('prefill_complete')))
    if is_ar:
        matches = [i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'timer.record']
        if len(matches) != 1:
            raise RuntimeError('AR prefill changed; review phase split')
        loop.body.insert(matches[0] + 1, ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id='generated', ctx=ast.Load())),
            body=[boundary], orelse=[]))
    else:
        predictions = [n for n in fn.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'pending_prediction' for t in n.targets)]
        if len(predictions) != 1:
            raise RuntimeError('PARD prefill changed; review phase split')
        # Like AR, selecting the first token belongs to decode, after prefill.
        prediction = predictions[0]
        fn.body.remove(prediction)
        index = fn.body.index(loop)
        fn.body[index:index] = [boundary, prediction]

    class RemoveInstrumentation(ast.NodeTransformer):
        removed = {'torch.cuda.reset_peak_memory_stats': 0, 'torch.cuda.synchronize': 0}

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                name = ast.unparse(node.value.func)
                if name in self.removed:
                    self.removed[name] += 1
                    return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            name = ast.unparse(node.func)
            if name in ('time.perf_counter', 'torch.cuda.max_memory_allocated'):
                return ast.copy_location(ast.Constant(0.0), node)
            return self.generic_visit(node)

    transform = RemoveInstrumentation()
    tree = ast.fix_missing_locations(transform.visit(tree))
    if transform.removed != {'torch.cuda.reset_peak_memory_stats': 1, 'torch.cuda.synchronize': 2}:
        raise RuntimeError('Internal instrumentation changed; review adapter')
    namespace = dict(vars(speculative), _StageTimer=UntimedStages)
    generated_source = ast.unparse(tree) + '\n'
    exec(compile(tree, '<benchmark-phased-' + fn.name + '>', 'exec'), namespace)
    return namespace[fn.name], generated_source


class PhasedRuntime:
    def __init__(self, runtime, prompt, tokens):
        if prompt.shape[0] != 1 or not runtime.ignore_eos:
            raise ValueError('Phased benchmark requires batch 1 and ignore_eos')
        if prompt.shape[1] + tokens + runtime.spec.draft_k > runtime.max_cache_len:
            raise ValueError('Insufficient verification lookahead capacity')
        self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
        method = (speculative.FusedPardRuntime._generate_ar if runtime.mode == 'ar'
                  else speculative.FusedPardRuntime._generate_spec)
        self.generator, self.generated_source = phased_generator(method)
        self.expected_ids = None
        self.checks = []

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def _target_call(self, ids, cache, positions, materialize_features=True):
        if cache.length == 0 and ids.shape[1] == self.prompt.shape[1]:
            if self.collector is not None:
                self.collector.reset()
            # Common prefill contract: project only the final position. TD
            # hooks still collect every prompt hidden state required by draft.
            output = self.target(input_ids=ids, past_key_values=cache,
                cache_position=positions, use_cache=True, attention_mask=None,
                return_dict=True, output_hidden_states=False, logits_to_keep=1)
            features = self.collector.features() if self.collector is not None else None
            return output, features
        return self.runtime._target_call(ids, cache, positions, materialize_features)

    def begin(self):
        state = self.generator(self, self.prompt, self.tokens)
        if next(state) != 'prefill_complete':
            raise RuntimeError('Missing prefill boundary')
        return state

    def finish(self, state):
        try:
            next(state)
        except StopIteration as stopped:
            return stopped.value
        raise RuntimeError('Unexpected extra generator boundary')

    def validate(self, result):
        ids = result.output_ids
        if len(ids) != self.tokens:
            raise RuntimeError('Wrong output length')
        if self.expected_ids is None:
            self.expected_ids = list(ids)
        if ids != self.expected_ids:
            raise RuntimeError('Output changed across phases/repeats')
        self.checks.append({'tokens': len(ids), 'exact': True,
                           'target_forwards': result.target_forwards,
                           'draft_forwards': result.draft_forwards,
                           'accepted_draft_tokens': result.accepted_draft_tokens,
                           'proposed_draft_tokens': result.proposed_draft_tokens,
                           'ar_fallback_tokens': getattr(result, 'ar_fallback_tokens', 0),
                           'fallback_after_verifier_steps': getattr(result, 'fallback_after_verifier_steps', None)})


class PrefillWorkload:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self):
        self.adapter.begin().close()


class DecodeWorkload:
    def __init__(self, adapter):
        self.adapter, self.state = adapter, None

    def prepare(self):
        if self.state is not None:
            self.state.close()
        self.state = self.adapter.begin()

    def __call__(self):
        self.adapter.validate(self.adapter.finish(self.state))
        self.state = None


class E2EWorkload:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self):
        self.adapter.validate(self.adapter.finish(self.adapter.begin()))


def run_phases(args):
    from e2e.benchmark_pard2 import preflight_gpu, capture_runtime_provenance, target_profile_preflight
    from e2e.pard2 import DEFAULT_DRAFT, DEFAULT_TOKENIZER
    if args.kv_cache_dtype != 'int4' or not args.int4_only:
        raise ValueError('Phased modes require --int4-only and INT4 KV')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_provenance = target_profile_preflight(args.int4_model, args.benchmark_profile)
    if args.benchmark_profile == 'qwen3_32b':
        # Preserve the documented 32B AR/TI numerical contract. Explicit
        # environment overrides remain available for separate experiments.
        os.environ.setdefault('QUAROT_FUSED_K1', '0')
        os.environ.setdefault('QUAROT_QWEN3_32B_GROUPED_NWAVES', '4')
        os.environ.setdefault('QUAROT_QWEN3_32B_MULTI_NWAVES', '2')
    gpu = preflight_gpu()
    torch.manual_seed(0)
    capacity = math.ceil((args.prefill_seq_len + args.decode_steps + 15) / 128) * 128
    runtime = speculative.load_runtime(mode=args.mode, target_checkpoint=args.int4_model,
        draft_snapshot=args.draft_model or str(DEFAULT_DRAFT),
        tokenizer_path=args.tokenizer or str(DEFAULT_TOKENIZER), max_cache_len=capacity,
        page_size=128, compile_mode='eager', ignore_eos=True, fused_norm_quant=True,
        benchmark_profile=args.benchmark_profile,
        ti_zero_accept_fallback=args.ti_zero_accept_fallback)
    runtime.target.cache_dtype = 'int4'
    runtime.draft_prefill_logits_to_keep = args.draft_prefill_logits
    linear_modules = harness.validate_real_int4_checkpoint(runtime.target)
    prompt = harness.deterministic_tokens(args.batch_size, args.prefill_seq_len, runtime.target.config.vocab_size, 'cuda')
    if args.input_ids:
        prompt = torch.tensor(json.loads(Path(args.input_ids).read_text()), dtype=torch.long, device='cuda')
        if tuple(prompt.shape) != (args.batch_size, args.prefill_seq_len):
            raise ValueError('Input JSON shape must match --batch-size and --prefill-seq-len')
        if not bool(((prompt >= 0) & (prompt < runtime.target.config.vocab_size)).all()):
            raise ValueError('Input token outside target vocabulary')
    if args.batch_size == 1:
        adapter = PhasedRuntime(runtime, prompt, args.decode_steps)
    elif args.mode == 'ar':
        from e2e.benchmark_batched_ar import BatchedAR
        adapter = BatchedAR(runtime, prompt, args.decode_steps)
    else:
        from e2e.benchmark_batched_pard import BatchedPARD
        adapter = BatchedPARD(runtime, prompt, args.decode_steps)
    provenance = capture_runtime_provenance(runtime, SimpleNamespace(compile_mode='eager', exact_small_chunk=None))
    mode_label = args.mode + '+ar-guard' if args.ti_zero_accept_fallback else args.mode
    workload_kind = 'provided_token_ids' if args.input_ids else 'synthetic_token_ids'
    if not args.input_ids:
        print('Workload: synthetic consecutive token IDs, without a chat template. '
              'Acceptance/speed apply to this stress input; use --input-ids for task-like contexts.', flush=True)
    payload = {'mode': mode_label, 'runtime_mode': args.mode, 'configuration': vars(args), 'gpu_preflight': gpu,
        'execution_policy': provenance['runtime_flags']['execution_policy'],
        'workload': {'kind': workload_kind,
                    'representative_task_speed': False if not args.input_ids else None,
                    'decoded_prompts': [runtime.tokenizer.decode(row) for row in prompt.cpu().tolist()]},
        'runtime_provenance': provenance, 'target_provenance': target_provenance,
        'input_ids': prompt.cpu().tolist(),
        'capacity': capacity, 'page_size': 128, 'linear4bit_modules': linear_modules,
        'protocol': 'shared teammate wall-time/memory harness; greedy fixed-length continuation; '
                    'prefill includes target/draft prefix; decode generates all requested tokens '
                    'from a prepared prefix including first proposal/verification; E2E performs both',
        'generated_adapter_source': adapter.generated_source,
        'benchmark_source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [('harness', harness.__file__), ('adapter', __file__)]},
        'phases': {}}
    if args.batch_size != 1:
        path = Path(inspect.getfile(type(adapter)))
        payload['benchmark_source_sha256']['batch_adapter'] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload['batch_cache_metadata'] = {
            'persistent_arange_indices': False,
            'static_decode_metadata': bool(runtime.verification_graph),
            'reason': 'Multi-page batched prefill requires interleaved page indices; use existing fallback.'}
    if args.mode != 'ar':
        from e2e import synchronous_pard
        payload['benchmark_source_sha256']['synchronous_runtime'] = hashlib.sha256(
            Path(synchronous_pard.__file__).read_bytes()).hexdigest()
        payload['batch_strategy'] = 'minimum accepted prefix' if args.batch_size > 1 else 'single sequence'
        payload['draft_prefill_logits_to_keep'] = args.draft_prefill_logits
    graphs = lambda: sorted(key[1] for key in runtime._verification_graphs)
    # Establish output parity and capture M15/M16 graphs before any phase is
    # measured, so resident graph/cache memory is common across all phases.
    with torch.inference_mode():
        if args.batch_size == 1:
            reference_ids = runtime.generate(prompt, args.decode_steps).output_ids
        else:
            reference_ids = [runtime._generate_ar(row[None], args.decode_steps).output_ids for row in prompt]
            # The serial oracle's batch-1 cache must not inflate batched peaks.
            runtime._verification_graphs.clear()
            runtime._verification_cache = None
            if runtime.collector is not None:
                runtime.collector.reset()
            harness.cleanup()
            payload['reference_kind'] = 'independent single-sequence target AR'
            if args.mode != 'ar':
                runtime_ids = runtime.generate(prompt, args.decode_steps).output_ids
                payload['runtime_batch_output_ids'] = runtime_ids
                if runtime_ids != reference_ids:
                    payload.update(qualification_failed=True, qualification_output_ids=runtime_ids,
                                   unmodified_runtime_output_ids=reference_ids)
                    output.write_text(json.dumps(payload, indent=2) + '\n')
                    raise RuntimeError('Batched runtime differs from target AR oracle')
        adapter.expected_ids = reference_ids
        payload['unmodified_runtime_output_ids'] = reference_ids
        for _ in range(2):
            result = adapter.finish(adapter.begin())
            try:
                adapter.validate(result)
            except RuntimeError:
                payload.update(qualification_failed=True, qualification_output_ids=result.output_ids)
                output.write_text(json.dumps(payload, indent=2) + '\n')
                raise
    payload['unmodified_runtime_output_ids'] = reference_ids
    payload['qualification_checks'] = list(adapter.checks)
    adapter.checks.clear()
    payload['graphs_before_measurement'] = graphs()
    for phase, workload in [('prefill', PrefillWorkload), ('decode', DecodeWorkload), ('e2e', E2EWorkload)]:
        payload['phases'][phase] = harness.benchmark_one(
            mode_label + ' ' + phase, lambda cls=workload: cls(adapter), args)
        payload['graphs_after_' + phase] = graphs()
        if graphs() != payload['graphs_before_measurement']:
            raise RuntimeError('Verification graph captured during phase measurement')
        output.write_text(json.dumps(payload, indent=2) + '\n')
    extension = provenance['hip_extension']
    extension_unchanged = hashlib.sha256(Path(extension['resolved_path']).read_bytes()).hexdigest() == extension['sha256']
    if not extension_unchanged:
        raise RuntimeError('HIP extension changed during benchmark')
    payload.update(output_ids=adapter.expected_ids, generation_checks=adapter.checks,
                   complete=True, extension_unchanged=extension_unchanged)
    if args.mode != 'ar' and adapter.checks and all(
            row['accepted_draft_tokens'] == 0 for row in adapter.checks):
        payload['performance_diagnostic'] = (
            'Zero accepted draft tokens: draft and chunk verification are overhead. '
            'This does not establish TI performance on natural task prompts.')
        print(payload['performance_diagnostic'], flush=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
