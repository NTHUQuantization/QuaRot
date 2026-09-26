"""Identical natural workloads with explicit phase memory and raw timing records."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from runtime_adapter import ROOT, WORKSPACE, BRANCHES, load, phases

PHASES=('prefill','decode','e2e')
DATASETS=('humaneval','gsm8k','math_500')
OLD=ROOT  # frozen prompt/order specification for this GPTQ run


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def append(path,obj):
    with path.open('a') as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def snapshot(repo):
    files=set()
    for dirname in ('quarot','e2e'):
        for p in (repo/dirname).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.so','.hip','.h','.cuh','.cpp'):
                files.add(p)
    files.add(repo/'setup.py')
    return {str(p.relative_to(repo)):sha(p) for p in sorted(files)}


def gpu_snapshot():
    result=subprocess.run(['rocm-smi','--showpids','--showuse','--showtemp','--showclocks','--showpower'],capture_output=True,text=True)
    processes=re.findall(r'^\s*(\d+)\s+\S+\s+(\d+)\s+(\d+)',result.stdout,re.M)
    gpu_processes=[p for p in processes if int(p[1])>0 or int(p[2])>0]
    return dict(unix=time.time(),returncode=result.returncode,text=result.stdout,
                stderr=result.stderr,gpu_processes=gpu_processes)


def difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)


def main(args):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'runs.jsonl').exists(), 'Refuse implicit restart/overwrite'
    started=time.perf_counter()
    free,total=torch.cuda.mem_get_info()
    assert total-free<1<<30, 'GPU is already occupied'
    source_before=snapshot(BRANCHES[args.variant])
    runtime=load(args.variant,out)
    loaded=time.perf_counter()
    import quarot
    sources=read_rows(OLD/'prompts.jsonl')
    common=json.loads((OLD/'protocol_manifest.json').read_text())
    selected={i:p for i,p in enumerate(sources) if p['stage']==args.stage}
    cap=128 if args.stage=='pilot' else 256
    adapters={i:phases(runtime,torch.tensor([p['input_ids']],dtype=torch.long,device='cuda'),cap)
              for i,p in selected.items()}
    reference={}
    oracle={}
    oracle_file=ROOT/'results'/args.stage/'D/runs.jsonl'
    if args.variant != 'D':
        assert (oracle_file.parent/'complete.json').exists(), 'Measure GPTQ D oracle first'
    for r in ([] if args.variant=='D' else read_rows(oracle_file)):
        if r['variant']=='D' and r['phase']=='e2e':
            oracle[r['dataset'],r['sample_id']]=r['output_ids']
    provenance=dict(variant=args.variant,stage=args.stage,
        source_commit=json.loads((ROOT/'host_git_metadata.json').read_text())[args.variant]['commit'],
        source_branch=json.loads((ROOT/'host_git_metadata.json').read_text())[args.variant]['branch'],
        source_sha256=source_before,extension_path=quarot._HIP.__file__,extension_sha256=sha(quarot._HIP.__file__),
        torch=torch.__version__,hip=torch.version.hip,python=sys.version,gpu=str(torch.cuda.get_device_properties(0)),
        load_seconds=loaded-started,eos_ids=sorted(runtime.eos_ids),cap=cap,repeats=3,
        prompt_file_sha256=sha(OLD/'prompts.jsonl'),oracle_sha256=None if args.variant=='D' else sha(oracle_file),
        adapter_sha256=sha(Path(__file__).with_name('runtime_adapter.py')),worker_sha256=sha(__file__),
        load_metadata=runtime.load_metadata,
        environment={k:v for k,v in os.environ.items() if k.startswith('QUAROT_') or k in ('OMP_NUM_THREADS','PYTHONPATH')},
        memory_contract='PyTorch peak allocated/reserved per phase, reset after initial synchronization and before timer; decode prefix outside timer and reset; resident weights/buffers included',
        numerical_contract='Same GPTQ checkpoint and IDs. ABC branch arithmetic retained; cross-oracle differences explicitly reported.')
    if args.variant in 'DE':
        from e2e.benchmark_pard2 import capture_runtime_provenance
        provenance['runtime_provenance']=capture_runtime_provenance(runtime,SimpleNamespace(compile_mode='eager',exact_small_chunk=None))
    else:
        provenance['runtime_provenance']=dict(fused_projection=False,fused_norm_quant=False,
            input_clip_ratio=0.9,kv_clip_ratio=1.0,kv_group_size=128,
            ffn_rotation='grouped_h256_v1',attention='flash_attention_2',
            cache_capacity=8192,page_size=128,cache_policy='shared persistent storage, fresh logical state',
            linear_implementation=type(runtime.target.model.layers[0].self_attn.q_proj).__name__)
    atomic(out/'variant_manifest.json',provenance)
    (out/'generated_adapter.py').write_text(next(iter(adapters.values())).generated_source)
    warm_started=time.perf_counter()
    # All eager prompt shapes are exercised before measuring any sample.
    for adapter in adapters.values():adapter.begin().close()
    representatives=[next(i for i,p in selected.items() if p['dataset']==d) for d in DATASETS]
    for phase in PHASES:
        for i in representatives:
            adapter=adapters[i]
            if phase=='prefill':adapter.begin().close()
            else:
                result=adapter.finish(adapter.begin())
                if i in reference:assert result.output_ids==reference[i], 'Warmup output changed'
                reference[i]=result.output_ids
    torch.cuda.synchronize()
    warm_seconds=time.perf_counter()-warm_started
    graphs=sorted(str(k) for k in runtime._verification_graphs)
    atomic(out/'warmup.json',dict(seconds=warm_seconds,initial_per_phase=3,shapes=len(adapters),graphs=graphs))
    measured=time.perf_counter();count=0;cross_failures=set()
    for sweep,order in enumerate(common['orders'][args.stage]):
        for i in order:
            if (ROOT/'STOP').exists():raise RuntimeError('STOP requested')
            p,adapter=selected[i],adapters[i]
            gpu=gpu_snapshot();append(out/'gpu.jsonl',dict(variant=args.variant,sweep=sweep,sample_id=p['sample_id'],**gpu))
            assert gpu['returncode']==0 and len(gpu['gpu_processes'])==1,'GPU competition/unreadable state'
            for phase in PHASES:
                state=adapter.begin() if phase=='decode' else None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                t=time.perf_counter()
                if phase=='prefill':result=adapter.begin()
                elif phase=='decode':result=adapter.finish(state)
                else:result=adapter.finish(adapter.begin())
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-t)*1000
                peak=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                row=dict(run_id=args.run_id,variant=args.variant,stage=args.stage,dataset=p['dataset'],
                    sample_id=p['sample_id'],input_hash=p['input_hash'],input_tokens=p['input_tokens'],
                    sweep=sweep,repeat=sweep,phase=phase,elapsed_ms=elapsed,
                    baseline_allocated_bytes=baseline,baseline_reserved_bytes=reserved,
                    peak_allocated_bytes=peak,peak_reserved_bytes=peak_reserved,
                    incremental_peak_allocated_bytes=peak-baseline,gpu_competition=False,
                    gpu_snapshot_unix=gpu['unix'],valid=True)
                if phase=='prefill':result.close()
                else:
                    ids=result.output_ids
                    stats=dataclasses.asdict(result) if dataclasses.is_dataclass(result) else vars(result)
                    stats={k:v for k,v in stats.items() if k not in ('output_ids','ttft_ms','steady_decode_ms','total_ms','peak_vram_bytes','stage_ms')}
                    row.update(output_ids=ids,output_tokens=len(ids),output_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        stop_reason='eos' if ids and ids[-1] in runtime.eos_ids else 'cap',cap_reached=len(ids)==cap,raw_stats=stats)
                    if i not in reference:reference[i]=ids
                    own=reference[i]==ids;expected=reference[i] if args.variant=='D' else oracle[p['dataset'],p['sample_id']]
                    cross=ids==expected
                    if not cross:cross_failures.add(i)
                    append(out/'qualification.jsonl',dict(variant=args.variant,dataset=p['dataset'],sample_id=p['sample_id'],
                        phase=phase,sweep=sweep,within_variant_exact=own,against_D_oracle_exact=cross,
                        first_difference_from_D=difference(expected,ids),expected_tokens=len(expected),actual_tokens=len(ids)))
                    row['against_D_oracle_exact']=cross
                    if not own or not ids or len(ids)>cap or any(t in runtime.eos_ids for t in ids[:-1]):
                        row.update(valid=False,invalid_reason='Within-variant/EOS/output-count violation')
                if sorted(str(k) for k in runtime._verification_graphs)!=graphs or elapsed<=0:
                    row.update(valid=False,invalid_reason='New graph capture or nonpositive time')
                append(out/'runs.jsonl',row);count+=1
                atomic(ROOT/'progress.json',dict(run_id=args.run_id,stage=args.stage,variant=args.variant,
                    completed=count,total=len(adapters)*9,elapsed_seconds=time.perf_counter()-measured,
                    sample_id=p['sample_id'],cross_oracle_mismatch_prompts=len(cross_failures)))
                assert row['valid'], row
            print(args.stage,args.variant,sweep,p['dataset'],p['sample_id'],count,'/',len(adapters)*9,flush=True)
    assert snapshot(runtime.repo)==source_before,'Runtime changed while measuring'
    atomic(out/'complete.json',dict(variant=args.variant,stage=args.stage,records=count,
        load_seconds=loaded-started,warmup_seconds=warm_seconds,
        measurement_wall_seconds=time.perf_counter()-measured,total_work_seconds=time.perf_counter()-started,
        prompts=len(adapters),cross_oracle_mismatch_prompts=len(cross_failures),
        all_within_variant_exact=True,finished_unix=time.time()))
    if hasattr(runtime,'close'):runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list('ABCDE'),required=True)
    p.add_argument('--stage',choices=('pilot','formal'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True)
    args=p.parse_args()
    import torch
    with torch.inference_mode():main(args)
