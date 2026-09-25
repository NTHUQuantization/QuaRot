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


PHASES=('prefill','decode','e2e')
DATASETS=('humaneval','gsm8k','math_500')



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
    for dirname in ('quarot','e2e','third-party'):
        for p in (repo/dirname).rglob('*'):
            if p.is_file() and 'build' not in p.parts and p.suffix in ('.py','.so','.hip','.h','.cuh','.cpp'):
                files.add(p)
    if (repo/'setup.py').exists():files.add(repo/'setup.py')
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


def main(args, config, adapter_module, contract):
    import torch
    ROOT = args.output
    OLD = Path(__file__).resolve().parent/'reference'
    load = adapter_module.load
    phases = adapter_module.phases
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'runs.jsonl').exists(), 'Refuse implicit restart/overwrite'
    started=time.perf_counter()
    free,total=torch.cuda.mem_get_info()
    assert total-free<1<<30, 'GPU is already occupied'
    source_before=snapshot(Path(config['repo']))
    runtime=load(config,out)
    loaded=time.perf_counter()
    sources=read_rows(OLD/'prompts.jsonl')
    common=json.loads((OLD/'profile.json').read_text())
    selected={i:p for i,p in enumerate(sources) if p['stage']==args.stage}
    cap=128 if args.stage=='pilot' else 256
    adapters={i:phases(runtime,torch.tensor([p['input_ids']],dtype=torch.long,device='cuda'),cap)
              for i,p in selected.items()}
    reference={}
    import importlib.metadata
    provenance=dict(label=config['label'],stage=args.stage,contract=contract,
        source_commit=config['source_commit'],source_branch=config['source_branch'],
        source_sha256=source_before,torch=torch.__version__,hip=torch.version.hip,
        python=sys.version,gpu=str(torch.cuda.get_device_properties(0)),
        host_cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), 'unknown'),
        dependencies={n:importlib.metadata.version(n) for n in ('transformers','safetensors','flash_attn')},
        load_seconds=loaded-started,eos_ids=sorted(runtime.eos_ids),cap=cap,repeats=3,
        prompt_file_sha256=sha(OLD/'prompts.jsonl'),
        adapter_sha256=sha(adapter_module.__file__),worker_sha256=sha(__file__),
        execution=adapter_module.provenance(runtime),
        environment={k:v for k,v in os.environ.items() if k.startswith('QUAROT_') or k in ('OMP_NUM_THREADS','PYTHONPATH')},
        memory_contract='PyTorch allocated/reserved phase peaks; resident weights included; not whole-device VRAM',
        numerical_contract='Within-run exact outputs required; compare.py reports cross-run output differences')
    atomic(out/'manifest.json',provenance)
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
    measured=time.perf_counter();count=0
    for sweep,order in enumerate(common['orders'][args.stage]):
        for i in order:
            if (ROOT/'STOP').exists():raise RuntimeError('STOP requested')
            p,adapter=selected[i],adapters[i]
            gpu=gpu_snapshot();append(out/'gpu.jsonl',dict(label=config['label'],sweep=sweep,sample_id=p['sample_id'],**gpu))
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
                row=dict(run_id=args.run_id,label=config['label'],stage=args.stage,dataset=p['dataset'],
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
                    own=reference[i]==ids
                    append(out/'qualification.jsonl',dict(label=config['label'],dataset=p['dataset'],sample_id=p['sample_id'],
                        phase=phase,sweep=sweep,within_run_exact=own))
                    if not own or not ids or len(ids)>cap or any(t in runtime.eos_ids for t in ids[:-1]):
                        row.update(valid=False,invalid_reason='Within-run/EOS/output-count violation')
                if sorted(str(k) for k in runtime._verification_graphs)!=graphs or elapsed<=0:
                    row.update(valid=False,invalid_reason='New graph capture or nonpositive time')
                append(out/'runs.jsonl',row);count+=1
                atomic(ROOT/'progress.json',dict(run_id=args.run_id,stage=args.stage,label=config['label'],
                    completed=count,total=len(adapters)*9,elapsed_seconds=time.perf_counter()-measured,
                    sample_id=p['sample_id']))
                assert row['valid'], row
            print(args.stage,config['label'],sweep,p['dataset'],p['sample_id'],count,'/',len(adapters)*9,flush=True)
    assert snapshot(runtime.repo)==source_before,'Runtime changed while measuring'
    atomic(out/'measurement_complete.json',dict(label=config['label'],stage=args.stage,records=count,
        load_seconds=loaded-started,warmup_seconds=warm_seconds,
        measurement_wall_seconds=time.perf_counter()-measured,total_work_seconds=time.perf_counter()-started,
        prompts=len(adapters),
        all_within_run_exact=True,finished_unix=time.time()))
    if hasattr(runtime,'close'):runtime.close()
