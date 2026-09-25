"""Explicit compatibility bridge; branch HIP kernels and quantized values stay intact."""
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_imports(config):
    repo = Path(config['repo'])
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(Path(config['fht_path'])))
    import quarot
    assert Path(quarot.__file__).resolve().is_relative_to(repo.resolve()), quarot.__file__
    assert Path(quarot._HIP.__file__).resolve().is_relative_to(repo.resolve()), quarot._HIP.__file__
    return repo


def cache_mask_sizes(self, cache_position, layer_idx):
    """Mask metadata for the existing paged cache under Transformers 4.57."""
    return self.length + cache_position.shape[0], 0


def load(settings, artifact_dir):
    import torch
    TARGET = Path(settings['target'])
    TOKENIZER = Path(settings['tokenizer'])
    DRAFT = settings.get('draft')
    engine = settings['engine']
    config_data = json.loads((TARGET/'config.json').read_text())
    assert config_data.get('quarot_conversion', {}).get('method') == 'gptq', 'GPTQ target required'
    assert config_data['quarot_ffn_format'] == 'grouped_h256_v1'
    repo = configure_imports(settings)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if engine == 'legacy-ar':
        # Transformers 4.57 Qwen3 passes plural past_key_values to attention.
        # These branches expect singular and otherwise silently ignore KV.
        import e2e.quantized_common as common
        from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit
        MultiLayerPagedKVCache4Bit.get_mask_sizes = cache_mask_sizes
        (artifact_dir/'cache_api_compatibility.py').write_text(inspect.getsource(cache_mask_sizes))
        original = Path(common.__file__).read_text()
        patched = original.replace(
            '        output_attentions = False\n',
            '        plural_cache = kwargs.pop("past_key_values", None)\n'
            '        if past_key_value is None:\n'
            '            past_key_value = plural_cache\n'
            '        output_attentions = False\n', 1)
        # The common checkpoint explicitly serializes this rotation matrix.
        needle = '            self.down_proj_quantizer = quarot.nn.Quantizer(input_clip)\n'
        assert patched.count(needle) == 1
        patched = patched.replace(needle,
            '            if self.down_proj_hadamard.had_rem_dim is not None:\n'
            '                self.down_proj_hadamard._non_persistent_buffers_set.discard("had_rem_dim")\n' + needle)
        assert original != patched
        (artifact_dir/'compatibility.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), patched.splitlines(True),
            fromfile='a/e2e/quantized_common.py', tofile='b/e2e/quantized_common.py')))
        (artifact_dir/'effective_quantized_common.py').write_text(patched)
        exec(compile(patched, str(common.__file__)+'[benchmark-compatibility]', 'exec'), vars(common))
        from e2e.model_registry import runtime_types
        config_cls, model_cls, _ = runtime_types(str(TARGET), local_files_only=True)
        config = config_cls.from_pretrained(str(TARGET), local_files_only=True,
            attn_implementation='flash_attention_2')
        # Explicit grouped-H256 rotation and clip=0.9 match the common GPTQ checkpoint.
        config.quarot_input_clip_ratio = 0.9
        config.quarot_kv_clip_ratio = 1.0
        config.quarot_kv_group_size = 128
        config.quarot_ffn_rotation = 'grouped_h256_v1'
        key_mapping = {
            r'\.mlp\.down_proj\.2\.': '.mlp.down_proj.',
            r'\.mlp\.down_proj\.0\.had_rem_dim': '.mlp.down_proj_hadamard.had_rem_dim',
        }
        model, loading = model_cls.from_pretrained(str(TARGET), config=config,
            torch_dtype=torch.float16, local_files_only=True,
            key_mapping=key_mapping, output_loading_info=True)
        assert not any(loading[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), loading
        model = model.eval().to('cuda')
        eos = config.eos_token_id
        runtime = SimpleNamespace(target=model, eos_ids=set(eos if isinstance(eos,list) else [eos]),
            mode='ar', _verification_graphs={}, _benchmark_cache=None,
            load_metadata=dict(key_mapping=key_mapping, loading=loading,
                effective_config=config.to_dict(), compatibility_sha256=hashlib.sha256(patched.encode()).hexdigest()))
    else:
        from e2e.speculative import load_runtime
        runtime = load_runtime(mode=engine,
            target_checkpoint=str(TARGET), draft_snapshot=str(DRAFT), tokenizer_path=str(TOKENIZER),
            benchmark_profile='qwen3_8b', max_cache_len=8192, page_size=128,
            compile_mode='eager', ignore_eos=False, adaptive_k=False,
            fused_norm_quant=True, ti_zero_accept_fallback=0)
        runtime.load_metadata = {}
    runtime.engine = engine
    runtime.repo = repo
    return runtime


class BaselinePhases:
    def __init__(self, runtime, prompt, tokens):
        self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
        self.generated_source = inspect.getsource(type(self))

    def generator(self):
        import torch
        runtime = self.runtime
        cache = runtime._benchmark_cache
        if cache is None:
            cache = runtime.target.build_cache(1, 128, 8192)
            runtime._benchmark_cache = cache
        # All branch cache request-local state is reset, not just length.
        cache.length = 0
        cache._needs_init = [True] * cache.n_layers
        ids = self.prompt
        positions = torch.arange(ids.shape[1], device=ids.device)
        output = runtime.target(input_ids=ids, past_key_values=cache,
            cache_position=positions, use_cache=True, attention_mask=None,
            return_dict=True, output_hidden_states=False, logits_to_keep=1)
        yield 'prefill_complete'
        generated = []
        while len(generated) < self.tokens:
            token = int(output.logits[:, -1].argmax(-1))
            generated.append(token)
            if token in runtime.eos_ids or len(generated) >= self.tokens:
                break
            ids = torch.tensor([[token]], dtype=self.prompt.dtype, device=self.prompt.device)
            positions = torch.arange(cache.length, cache.length+1, device=ids.device)
            output = runtime.target(input_ids=ids, past_key_values=cache,
                cache_position=positions, use_cache=True, attention_mask=None,
                return_dict=True, output_hidden_states=False, logits_to_keep=1)
            cache = output.past_key_values
        return SimpleNamespace(output_ids=generated, target_forwards=len(generated),
            draft_forwards=0, proposed_draft_tokens=0, accepted_draft_tokens=0,
            verifier_steps=0, emitted_tokens_per_step=[1]*len(generated),
            ar_fallback_tokens=0, fallback_after_verifier_steps=None)

    def begin(self):
        state = self.generator()
        assert next(state) == 'prefill_complete'
        return state

    def finish(self, state):
        try:
            next(state)
        except StopIteration as stopped:
            return stopped.value
        raise RuntimeError('Unexpected additional phase boundary')


def phases(runtime, prompt, tokens):
    if runtime.engine == 'legacy-ar':
        return BaselinePhases(runtime, prompt, tokens)
    from e2e.benchmark_pard2_phases import PhasedRuntime, phased_generator
    from e2e.speculative import FusedPardRuntime
    class NaturalPhases(PhasedRuntime):
        def __init__(self):
            self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
            method = FusedPardRuntime._generate_ar if runtime.mode=='ar' else FusedPardRuntime._generate_spec
            self.generator, self.generated_source = phased_generator(method)
    return NaturalPhases()



def provenance(runtime):
    import quarot
    import fast_hadamard_transform as fht
    details = dict(extension_path=quarot._HIP.__file__,
                   extension_sha256=sha(quarot._HIP.__file__),
                   fht_path=fht.__file__,fht_sha256=sha(fht.__file__),
                   load_metadata=runtime.load_metadata)
    if runtime.engine != 'legacy-ar':
        from e2e.benchmark_pard2 import capture_runtime_provenance
        details['runtime'] = capture_runtime_provenance(runtime, SimpleNamespace(compile_mode='eager',exact_small_chunk=None))
    else:
        details['runtime'] = dict(input_clip_ratio=0.9,kv_clip_ratio=1.0,kv_group_size=128,
            ffn_rotation='grouped_h256_v1',attention='flash_attention_2',
            cache_capacity=8192,page_size=128,mode='ar',
            linear_implementation=type(runtime.target.model.layers[0].self_attn.q_proj).__name__)
    return details
