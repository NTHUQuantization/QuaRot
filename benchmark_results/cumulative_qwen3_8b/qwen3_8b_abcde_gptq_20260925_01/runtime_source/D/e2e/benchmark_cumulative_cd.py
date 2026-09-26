"""Natural-EOS, paired C/D benchmark for the 2026-09-25 Chinese protocol.

Run in the existing ROCm environment; results never reuse historical timings.
The AST phase splitter is shared with benchmark_pard2_phases, but correctness
and serialization occur strictly outside synchronized wall-time boundaries.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))
DATA = REPO / 'qwen3_full_eval_20260925/data'
TARGET = WORKSPACE / 'qwen3_8b_fused_v1_rtn_w4a4kv4'
CACHE = WORKSPACE / '.hf_cache/pard/hub'
DRAFT = CACHE / 'models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135'
TOKENIZER = CACHE / 'models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
DATASETS = ('humaneval', 'gsm8k', 'math_500')
PHASES = ('prefill', 'decode', 'e2e')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def sources():
    paths = set((REPO / 'quarot').rglob('*.py'))
    paths.update((REPO / 'quarot').glob('*.so'))
    paths.update((REPO / 'e2e').glob('*.py'))
    for p in (REPO / 'e2e').glob('quantized*'):
        if p.is_dir():
            paths.update(p.rglob('*.py'))
    paths.update((REPO / 'third-party/hadacore').glob('*.so'))
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def prepare(out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    prompts = []
    for dataset in DATASETS:
        path = DATA / (dataset + '.jsonl')
        assert sha(path) == manifest[dataset]['prepared_sha256']
        data = rows(path)
        assert len(data) == manifest[dataset]['count']
        assert len({x['id'] for x in data}) == len(data)
        # Freeze selection before seeing any generation/acceptance/timing.
        indices = random.Random(0).sample(range(len(data)), 36)
        for i, index in enumerate(indices):
            item = data[index]
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': item['data']}]
            ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          enable_thinking=False)
            assert len(ids) + 256 + 15 <= 8192, 'No truncation or silent exclusions'
            prompts.append(dict(dataset=dataset, sample_id=item['id'], source_index=index,
                stage='formal' if i < 32 else 'pilot', input_ids=ids,
                input_tokens=len(ids), input_hash=digest(ids), messages=messages,
                prompt_hash=digest(messages), template_sha256=digest(tok.chat_template),
                source=manifest[dataset]))
    for p in prompts:
        append(out / 'prompts.jsonl', p)
    orders = {}
    for stage in ('pilot', 'formal'):
        indices = [i for i,p in enumerate(prompts) if p['stage'] == stage]
        orders[stage] = []
        for sweep in range(3):
            order = indices.copy()
            random.Random(sweep).shuffle(order)
            orders[stage].append(order)
    write(out / 'protocol_manifest.json', dict(run_id=out.name, created_unix=time.time(),
        protocol_sha256=sha(WORKSPACE / 'QWEN3_8B_CUMULATIVE_ABLATION_PROTOCOL_ZH.md'),
        scope=['C', 'D'], sampling='random.Random(0).sample(source-order indices, 36); first 32 formal, last 4 pilot',
        orders=orders, repeats=3, batch_size=1, pilot_max_new_tokens=128,
        formal_max_new_tokens=256, seed=0, bootstrap_seed=0, bootstrap_repeats=2000,
        ignore_eos=False, enable_thinking=False, greedy=True, stop='target config EOS only; include committed EOS',
        system_prompt='You are a helpful assistant.', tokenizer=str(TOKENIZER),
        cache_capacity=8192, page_size=128, target_dtype='float16', kv_dtype='int4',
        attention_backend='flash_attention_2 target; eager BF16 draft',
        compile_mode='eager', verification_graph=True, draft_k=15,
        adaptive_k=False, ar_fallback=0, draft_dtype='bfloat16', draft=str(DRAFT),
        initial_warmups_per_phase=3,
        warmup_policy='one representative per dataset per phase; prefill each input shape outside measurements; prewarm verifier M15/M16; invalidate newly captured graph samples',
        timing='perf_counter + GPU synchronize only at phase boundaries; decode prefix prepared outside timer; E2E no boundary synchronization; validation/serialization excluded',
        ordering='each stage C then D, each process loaded once for 3 sweeps; model residency amortized, possible clock drift recorded',
        invalidity='GPU competition, nonpositive timing, empty output, or new graph capture invalidates sample and halts; no latency-based deletion',
        sources=sources(), data_sources=manifest, input_policy='no truncation; fail if input+256+15 exceeds 8192',
        weight_files={str(p): sha(p) for base in (TARGET, DRAFT) for p in sorted(base.iterdir())
                      if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin')},
        environment={k:v for k,v in os.environ.items() if k.startswith('QUAROT_') or k in ('OMP_NUM_THREADS','PYTHONPATH')}))


def gpu_snapshot():
    result = subprocess.run(['rocm-smi', '--showpids', '--showuse', '--showtemp', '--showclocks', '--showpower'],
                            capture_output=True, text=True)
    return dict(unix=time.time(), returncode=result.returncode, text=result.stdout, stderr=result.stderr)


def run(out, stage, variant):
    import torch
    from e2e.speculative import load_runtime, FusedPardRuntime
    from e2e.benchmark_pard2_phases import PhasedRuntime, phased_generator
    from e2e.benchmark_pard2 import capture_runtime_provenance, preflight_gpu, target_profile_preflight
    class Adapter(PhasedRuntime):
        def __init__(self, runtime, prompt, tokens):
            self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
            method = FusedPardRuntime._generate_ar if runtime.mode == 'ar' else FusedPardRuntime._generate_spec
            self.generator, self.generated_source = phased_generator(method)

    manifest = json.loads((out / 'protocol_manifest.json').read_text())
    assert sources() == manifest['sources'], 'Source changed since freeze'
    path = out / stage
    path.mkdir(exist_ok=True)
    assert not any(r['variant'] == variant for r in rows(path / 'runs.jsonl')), 'Use a new run directory, no implicit resume'
    torch.manual_seed(0)
    start = time.perf_counter()
    preflight = preflight_gpu()
    mode = 'ar' if variant == 'C' else 'pard2-ti'
    target_contract = target_profile_preflight(TARGET, 'qwen3_8b')
    runtime = load_runtime(mode=mode, target_checkpoint=str(TARGET), draft_snapshot=str(DRAFT),
        tokenizer_path=str(TOKENIZER), max_cache_len=8192, page_size=128, compile_mode='eager',
        ignore_eos=False, fused_norm_quant=True, benchmark_profile='qwen3_8b',
        adaptive_k=False, ti_zero_accept_fallback=0)
    load_seconds = time.perf_counter() - start
    assert runtime.target.cache_dtype == 'int4' and runtime.spec.draft_k == 15
    assert not runtime.ignore_eos and runtime.adaptive_k is None
    provenance = capture_runtime_provenance(runtime, SimpleNamespace(compile_mode='eager', exact_small_chunk=None))
    append(out / 'variant_manifest.jsonl', dict(stage=stage, variant=variant, mode=mode,
        target=str(TARGET), draft=str(DRAFT) if variant == 'D' else None,
        spec=dataclasses.asdict(runtime.spec), eos_ids=sorted(runtime.eos_ids),
        provenance=provenance, target_contract=target_contract, preflight=preflight,
        torch=torch.__version__, hip=torch.version.hip, python=sys.version,
        gpu_properties=str(torch.cuda.get_device_properties(0)), load_seconds=load_seconds))
    prompts = rows(out / 'prompts.jsonl')
    tokens = 128 if stage == 'pilot' else 256
    adapters = {i: Adapter(runtime, torch.tensor([p['input_ids']], dtype=torch.long, device='cuda'), tokens)
                for i,p in enumerate(prompts) if p['stage'] == stage}
    sample = next(iter(adapters.values()))
    (path / (variant + '_generated_adapter.py')).write_text(sample.generated_source)
    warm_start = time.perf_counter()
    reference = {}
    def invoke(adapter, phase):
        if phase == 'prefill':
            state = adapter.begin()
            return state
        if phase == 'decode':
            return adapter.finish(adapter.begin())
        return adapter.finish(adapter.begin())
    # Eager shape initialization and graph captures are excluded from samples.
    for adapter in adapters.values():
        adapter.begin().close()
    representatives = [next(i for i in adapters if prompts[i]['dataset'] == d) for d in DATASETS]
    for phase in PHASES:
        for i in representatives:
            result = invoke(adapters[i], phase)
            if phase == 'prefill':
                result.close()
            else:
                if i in reference:
                    assert result.output_ids == reference[i]
                reference[i] = result.output_ids
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    expected_graphs = sorted(str(k) for k in runtime._verification_graphs)
    write(path / (variant + '_warmup.json'), dict(seconds=warm_seconds, graphs=expected_graphs,
        initial_per_phase=3, prefill_shapes=len(adapters)))
    measured_start = time.perf_counter()
    total = len(adapters) * 3 * 3
    count = 0
    with torch.inference_mode():
        for sweep, order in enumerate(manifest['orders'][stage]):
            for i in order:
                if (out / 'STOP').exists():
                    raise RuntimeError('STOP requested')
                p, adapter = prompts[i], adapters[i]
                gpu = gpu_snapshot()
                append(path / 'gpu.jsonl', dict(variant=variant, sweep=sweep, sample_id=p['sample_id'], **gpu))
                # Runtime PID is the only allowed GPU compute process.
                import re
                processes = re.findall(r'^\s*(\d+)\s+\S+\s+(\d+)\s+(\d+)', gpu['text'], re.M)
                gpu_pids = [pid for pid, devices, vram in processes if int(devices) > 0 or int(vram) > 0]
                if gpu['returncode'] != 0 or len(gpu_pids) != 1:
                    raise RuntimeError('GPU process state unreadable or competing process: ' + gpu['text'])
                for phase in PHASES:
                    state = adapter.begin() if phase == 'decode' else None
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    if phase == 'prefill':
                        result = adapter.begin()
                    elif phase == 'decode':
                        result = adapter.finish(state)
                    else:
                        result = adapter.finish(adapter.begin())
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - began) * 1000
                    record = dict(run_id=out.name, stage=stage, variant=variant, dataset=p['dataset'],
                        sample_id=p['sample_id'], input_hash=p['input_hash'], input_tokens=p['input_tokens'],
                        sweep=sweep, repeat=sweep, phase=phase, elapsed_ms=elapsed,
                        gpu_competition=False, gpu_snapshot_unix=gpu['unix'], valid=True)
                    if phase == 'prefill':
                        result.close()
                    else:
                        ids = result.output_ids
                        stats = {k:v for k,v in dataclasses.asdict(result).items()
                                 if k not in ('ttft_ms','steady_decode_ms','total_ms','stage_ms','peak_vram_bytes','output_ids')}
                        reason = 'eos' if ids and ids[-1] in runtime.eos_ids else 'cap'
                        record.update(output_ids=ids, output_hash=digest(ids), output_tokens=len(ids),
                            stop_reason=reason, cap_reached=len(ids) == tokens,
                            raw_stats=stats, actual_emitted_tokens=len(ids))
                        if not ids or len(ids) > tokens or any(t in runtime.eos_ids for t in ids[:-1]):
                            record.update(valid=False, invalid_reason='output count/EOS contract')
                        if i not in reference:
                            reference[i] = ids
                        same = ids == reference[i]
                        append(path / 'qualification.jsonl', dict(kind='within_variant', variant=variant,
                            dataset=p['dataset'], sample_id=p['sample_id'], sweep=sweep, phase=phase,
                            exact=same, first_difference=first_difference(reference[i], ids)))
                        if not same:
                            record.update(valid=False, invalid_reason='within-variant output mismatch')
                    graphs = sorted(str(k) for k in runtime._verification_graphs)
                    if elapsed <= 0 or graphs != expected_graphs:
                        record.update(valid=False, invalid_reason='nonpositive time or new graph capture')
                    append(path / 'runs.jsonl', record)
                    count += 1
                    write(out / 'progress.json', dict(stage=stage, variant=variant, completed=count,
                        total=total, elapsed_seconds=time.perf_counter()-measured_start,
                        dataset=p['dataset'], sample_id=p['sample_id']))
                    if not record['valid']:
                        raise RuntimeError(record['invalid_reason'])
                print(stage, variant, sweep, p['dataset'], p['sample_id'], count, '/', total, flush=True)
    assert sources() == manifest['sources'], 'Source changed during measurement'
    write(path / (variant + '_complete.json'), dict(load_seconds=load_seconds,
        warmup_seconds=warm_seconds, measurement_wall_seconds=time.perf_counter()-measured_start,
        process_work_seconds=time.perf_counter()-start, records=count, completed_unix=time.time()))
    runtime.close()


def first_difference(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x != y),
                min(len(a),len(b)) if len(a) != len(b) else None)


def csv_write(path, records):
    if records:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def summarize(out, stage):
    path = out / stage
    raw = rows(path / 'runs.jsonl')
    prompts = [p for p in rows(out / 'prompts.jsonl') if p['stage'] == stage]
    grouped = {}
    for r in raw:
        grouped.setdefault((r['variant'], r['dataset'], r['sample_id'], r['phase']), []).append(r)
    per = []
    for (v,d,i,p), rr in grouped.items():
        per.append(dict(variant=v, dataset=d, sample_id=i, phase=p,
            repeats=len(rr), elapsed_ms=statistics.median(x['elapsed_ms'] for x in rr),
            tokens_s=statistics.median(1000*x['output_tokens']/x['elapsed_ms'] for x in rr) if p != 'prefill' else '',
            output_tokens=rr[0].get('output_tokens',''), cap_reached=rr[0].get('cap_reached','')))
    csv_write(path / 'per_prompt.csv', per)
    lookup = {(r['variant'],r['dataset'],r['sample_id'],r['phase']):r for r in per}
    checks = []
    for p in prompts:
        key = (p['dataset'],p['sample_id'])
        a = grouped.get(('C',*key,'e2e'),[])
        b = grouped.get(('D',*key,'e2e'),[])
        complete = all(len(grouped.get((v,*key,phase),[])) == 3 for v in ('C','D') for phase in PHASES)
        exact = bool(a and b) and a[0]['output_ids'] == b[0]['output_ids'] and a[0]['stop_reason'] == b[0]['stop_reason']
        checks.append(dict(kind='cross_variant', dataset=key[0], sample_id=key[1], complete=complete,
            exact=exact, first_difference=first_difference(a[0]['output_ids'],b[0]['output_ids']) if a and b else None,
            C_output_tokens=a[0]['output_tokens'] if a else None, D_output_tokens=b[0]['output_tokens'] if b else None))
    (path / 'cross_variant_qualification.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in checks))
    qualified = all(c['exact'] and c['complete'] for c in checks) and all(r['valid'] for r in raw)
    summary = []
    speeds = []
    rng = random.Random(0)
    dataset_logs = []
    for d in DATASETS:
        for v in ('C','D'):
            result = dict(dataset=d, variant=v)
            for phase in PHASES:
                rr = [r for r in per if r['dataset']==d and r['variant']==v and r['phase']==phase]
                result[phase+'_ms'] = statistics.median(r['elapsed_ms'] for r in rr) if rr else None
                if phase != 'prefill':
                    result[phase+'_tokens_s'] = statistics.median(r['tokens_s'] for r in rr) if rr else None
            rr = [r for r in per if r['dataset']==d and r['variant']==v and r['phase']=='e2e']
            result['coverage'] = len(rr)
            result['cap_reached_fraction'] = statistics.mean(r['cap_reached'] for r in rr) if rr else None
            result['qualified_same_output'] = qualified
            summary.append(result)
        logs=[]
        for p in prompts:
            if p['dataset'] != d: continue
            a=lookup.get(('C',d,p['sample_id'],'e2e'))
            b=lookup.get(('D',d,p['sample_id'],'e2e'))
            if a and b: logs.append(math.log(a['elapsed_ms']/b['elapsed_ms']))
        if len(logs)==sum(p['dataset']==d for p in prompts):
            boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(2000))
            speeds.append(dict(dataset=d, paired_e2e_C_to_D=math.exp(statistics.mean(logs)),
                ci95_low=boot[49], ci95_high=boot[1949], prompts=len(logs), qualified_same_output=qualified))
            dataset_logs.append(logs)
    if len(dataset_logs)==3:
        boot=sorted(math.exp(statistics.mean(statistics.mean(rng.choices(x,k=len(x))) for x in dataset_logs)) for _ in range(2000))
        speeds.append(dict(dataset='overall_equal_dataset',paired_e2e_C_to_D=math.exp(statistics.mean(statistics.mean(x) for x in dataset_logs)),
            ci95_low=boot[49],ci95_high=boot[1949],prompts=len(prompts),qualified_same_output=qualified))
    csv_write(path / 'summary_by_dataset.csv',summary)
    csv_write(path / 'speedups.csv',speeds)
    write(path / 'qualification_summary.json', dict(qualified=qualified,prompts=len(prompts),
        exact=sum(c['exact'] for c in checks),complete=sum(c['complete'] for c in checks),
        mismatches=[c for c in checks if not c['exact']],raw_records=len(raw)))
    lines=[f'# Qwen3-8B C／D {stage} 測量', '',
        f'資格：{"通過 C／D 同輸出驗證" if qualified else "尚未通過；以下僅供診斷，不可解讀為同輸出純執行加速"}。',
        f'共 {len(prompts)} 題；每題每 phase 3 次；batch=1；自然 EOS；thinking 關閉；輸出上限 {128 if stage=="pilot" else 256} tokens。',
        'C：融合 INT4 AR；D：相同 target + PARD2-TI，BF16 draft，k=15，無 adaptive-k／AR fallback。',
        'E2E 是本地生成時間，不含載入、tokenizer、網路、輸出比對。先對 repeats 取 median，再對 prompts 取 median。', '',
        '|資料集|版本|Prefill ms|Decode ms|E2E ms|Decode tokens/s|E2E tokens/s|cap 比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        vals=[r[k] for k in ('prefill_ms','decode_ms','e2e_ms','decode_tokens_s','e2e_tokens_s','cap_reached_fraction')]
        lines.append('|'+r['dataset']+'|'+r['variant']+'|'+'|'.join(f'{v:.3f}' if v is not None else '缺失' for v in vals)+'|')
    lines += ['', '|資料集|C→D E2E 配對幾何平均|95% bootstrap CI|','|---|---:|---:|']
    for r in speeds:
        lines.append(f'|{r["dataset"]}|{r["paired_e2e_C_to_D"]:.3f}×|{r["ci95_low"]:.3f}–{r["ci95_high"]:.3f}|')
    lines += ['', f'跨版本輸出完全相同：{sum(c["exact"] for c in checks)}/{len(prompts)}；完整量測：{sum(c["complete"] for c in checks)}/{len(prompts)}。',
        'A／B 未在本次範圍內測量，故不報相對 A 加速，也不宣稱完成四版本消融。',
        '本次單次載入每版本後完成 3 sweeps；GPU 時脈／溫度／process 狀態見 gpu.jsonl。',
        '不一致樣本全部保留於 cross_variant_qualification.jsonl；未刪除或以成功交集替代。']
    for v in ('C','D'):
        p=path/(v+'_complete.json')
        if p.exists(): lines += ['',v+' 成本（秒）：`'+json.dumps(json.loads(p.read_text()))+'`']
    lengths={}
    for d in DATASETS:
        lengths[d]={'input_tokens':sorted(p['input_tokens'] for p in prompts if p['dataset']==d)}
        for v in ('C','D'):
            lengths[d][v+'_output_tokens']=sorted(r['output_tokens'] for r in per if r['dataset']==d and r['variant']==v and r['phase']=='e2e')
    write(path/'token_lengths.json',lengths)
    (path/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=stage,qualified=qualified,exact=sum(c['exact'] for c in checks),prompts=len(prompts))),flush=True)
    return qualified


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('prepare','run','summarize','all'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('pilot','formal'),default='pilot')
    p.add_argument('--variant',choices=('C','D'))
    args=p.parse_args()
    if args.action=='prepare': prepare(args.output)
    elif args.action=='run':
        import torch
        with torch.inference_mode(): run(args.output,args.stage,args.variant)
    elif args.action=='summarize': summarize(args.output,args.stage)
    else:
        started=time.time()
        if not args.output.exists(): prepare(args.output)
        for stage in ('pilot','formal'):
            for v in ('C','D'):
                if (args.output/stage/(v+'_complete.json')).exists():
                    continue
                cmd=[sys.executable,str(Path(__file__)), 'run','--output',str(args.output),'--stage',stage,'--variant',v]
                append(args.output/'commands.jsonl',dict(unix=time.time(),command=cmd))
                with (args.output/f'{stage}_{v}.log').open('x') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not summarize(args.output,stage):
                raise RuntimeError(stage+' output qualification failed; retained all diagnostics; formal progression blocked')
        write(args.output/'complete.json',dict(wall_seconds=time.time()-started,finished_unix=time.time()))


if __name__=='__main__':
    main()
