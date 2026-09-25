"""Measure one user-named implementation under a frozen Qwen3-8B profile."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ENV = dict(OMP_NUM_THREADS='4', QUAROT_BATCHED_H128='1',
           QUAROT_STATIC_KV_METADATA='1', QUAROT_VERIFICATION_GRAPH='1',
           QUAROT_CHUNK_PREPROCESS='1', PYTHONDONTWRITEBYTECODE='1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def read_config(path):
    path = path.resolve()
    config = json.loads(path.read_text())
    allowed = {'label','run_id','repo','engine','target','draft','tokenizer','output','fht_path','adapter'}
    if set(config)-allowed:
        raise ValueError(f'Unknown config keys: {set(config)-allowed}; fixed profile cannot be overridden')
    for key in ('target','draft','tokenizer','output','repo','fht_path','adapter'):
        if config.get(key):config[key] = str((path.parent/Path(config[key]).expanduser()).resolve())
    for key in ('target','tokenizer','output','repo','label','run_id','engine'):
        if not config.get(key):raise ValueError(f'Missing {key}')
    for key in ('label','run_id'):
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+',config[key]):raise ValueError(f'{key}: use letters, digits, dot, dash, underscore')
    if not config.get('adapter') and config['engine'] not in ('ar','pard2-ti','legacy-ar'):
        raise ValueError('Choose a built-in engine or supply a custom adapter')
    if config['engine']=='pard2-ti' and not config.get('draft'):
        raise ValueError('pard2-ti requires the pinned draft checkpoint')
    config.setdefault('fht_path',str(Path(config['repo'])/'third-party/hadacore'))
    return config


def git(repo, *args):
    return subprocess.check_output(['git','-c','safe.directory='+str(repo),'-C',str(repo),*args],text=True).strip()


def verify(config):
    if sys.flags.optimize:raise RuntimeError('python -O disables required validation')
    from require_gptq import require_gptq_checkpoint
    require_gptq_checkpoint(config['target'])
    expected = json.loads((HERE/'reference/input_hashes.json').read_text())
    actual = {}
    for kind in ('target','tokenizer','draft'):
        if kind=='draft' and not config.get('draft'):continue
        actual[kind]={}
        for name,wanted in expected[kind].items():
            p=Path(config[kind])/name
            actual[kind][name]=digest(p)
            if actual[kind][name]!=wanted:raise ValueError(f'Frozen input checksum mismatch: {p}')
    if not config.get('adapter'):
        if not list((Path(config['repo'])/'quarot').glob('_HIP*.so')):
            raise ValueError('Build your selected checkout extension in-place first')
        if not list(Path(config['fht_path']).glob('fast_hadamard_transform*.so')):
            raise ValueError('Build the FHT dependency or configure fht_path')
    git(config['repo'],'rev-parse','HEAD')
    return actual


def contract(config, actual):
    common = dict(profile_sha256=digest(HERE/'reference/profile.json'),
        prompts_sha256=digest(HERE/'reference/prompts.jsonl'),
        target_sha256=actual['target'],tokenizer_sha256=actual['tokenizer'],
        environment=ENV,worker_sha256=digest(HERE/'measure.py'))
    return dict(comparison_id=identity(common),common=common,inputs=actual,
                config=config,adapter_sha256=digest(config.get('adapter') or HERE/'runtime_adapter.py'))


def load_adapter(config):
    path=Path(config.get('adapter') or HERE/'runtime_adapter.py')
    spec=importlib.util.spec_from_file_location('selected_benchmark_adapter',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    for name in ('load','phases','provenance'):
        if not callable(getattr(module,name,None)):raise ValueError(f'Adapter requires {name}()')
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('pilot','formal'),default='pilot')
    parser.add_argument('--check',action='store_true',help='CPU input/checkpoint/build checks only')
    args=parser.parse_args()
    config=read_config(args.config)
    output=Path(config['output'])/config['run_id']/args.stage
    if not args.check and output.exists():raise ValueError(f'Refuse overwrite/partial resume: {output}')
    actual=verify(config)
    if args.check:
        print('GPTQ + exact target/tokenizer/draft hashes + selected checkout/build files verified. No GPU loaded.')
        return
    unexpected=[k for k in os.environ if k.startswith('QUAROT_') and k not in ENV]
    if unexpected:raise ValueError(f'Unset non-profile runtime tuning variables: {unexpected}')
    for key,value in ENV.items():
        if os.environ.get(key,value)!=value:raise ValueError(f'{key} must be {value}')
        os.environ[key]=value
    repo=Path(config['repo'])
    config.update(source_commit=git(repo,'rev-parse','HEAD'),source_branch=git(repo,'branch','--show-current'))
    gitroot=git(repo,'rev-parse','--show-toplevel')
    count=int(os.environ.get('GIT_CONFIG_COUNT','0'))
    os.environ.update({f'GIT_CONFIG_KEY_{count}':'safe.directory',f'GIT_CONFIG_VALUE_{count}':gitroot,
                       'GIT_CONFIG_COUNT':str(count+1)})
    bound=contract(config,actual)
    output.mkdir(parents=True,exist_ok=False)
    (output/'contract.json').write_text(json.dumps(bound,indent=2)+'\n')
    (output/'prompts.jsonl').write_bytes((HERE/'reference/prompts.jsonl').read_bytes())
    (output/'profile.json').write_bytes((HERE/'reference/profile.json').read_bytes())
    import measure
    adapter=load_adapter(config)
    args.output=output
    args.run_id=config['run_id']
    import torch
    with torch.inference_mode():measure.main(args,config,adapter,bound)
    # Recheck inputs outside all timed sections. Failure invalidates this run.
    try:
        if verify(config)!=actual:raise ValueError('Inputs changed while measuring')
        if digest(adapter.__file__)!=bound['adapter_sha256']:raise ValueError('Adapter changed while measuring')
        for name,key in [('measure.py','worker_sha256'),('reference/profile.json','profile_sha256'),('reference/prompts.jsonl','prompts_sha256')]:
            if digest(HERE/name)!=bound['common'][key]:raise ValueError(f'Benchmark contract changed while measuring: {name}')
    except Exception:
        (output/'complete.json').unlink(missing_ok=True)
        raise
    (output/'measurement_complete.json').replace(output/'complete.json')
    import summarize
    summarize.main(output)


if __name__=='__main__':main()
