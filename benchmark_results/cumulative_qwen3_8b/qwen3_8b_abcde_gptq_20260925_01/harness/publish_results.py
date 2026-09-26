"""Audit all artifacts, package results, then fast-forward only result commits."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
RUN_ID='qwen3_8b_abcde_gptq_20260925_01'
RESULT_PATH=Path('benchmark_results/cumulative_qwen3_8b')/RUN_ID
from runtime_adapter import BRANCHES as REPOS
URLS={'ABC':'git@github.com:NTHUQuantization/QuaRot_Version.git',
      'DE':'git@github.com:NTHUQuantization/QuaRot.git'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)


def audit():
    from verify_source_migration import verify
    allowed_adapter_hashes=verify(ROOT)
    for stage in ('pilot','formal'):
        subprocess.run([sys.executable,str(ROOT/'summarize.py'),stage],check=True)
    protocol=json.loads((ROOT/'protocol_manifest.json').read_text())
    verified={}
    for name,digest in protocol['weight_files'].items():
        path=Path(name.replace('/workspace_root',str(WORKSPACE),1))
        assert sha(path)==digest, f'Checkpoint changed: {path}'
        verified[name]=digest
    for name,digest in protocol['tokenizer_files_sha256'].items():
        path=Path(protocol['tokenizer'].replace('/workspace_root',str(WORKSPACE),1))/name
        assert sha(path)==digest, f'Tokenizer changed: {path}'
    for v,repo in REPOS.items():
        m=json.loads((ROOT/'results/formal'/v/'variant_manifest.json').read_text())
        pilot=json.loads((ROOT/'results/pilot'/v/'variant_manifest.json').read_text())
        assert m['source_commit']==git(repo,'rev-parse','HEAD')
        assert m['source_sha256']==pilot['source_sha256']
        assert sha(ROOT/'prompts.jsonl')==m['prompt_file_sha256']==pilot['prompt_file_sha256']
        for stage,record in [('pilot',pilot),('formal',m)]:
            if v != 'D':
                assert sha(ROOT/'results'/stage/'D/runs.jsonl')==record['oracle_sha256']
        for name,digest in m['source_sha256'].items():
            assert sha(repo/name)==digest, f'Source changed: {v}/{name}'
        assert sha(ROOT/'measure.py')==m['worker_sha256']==pilot['worker_sha256']
        assert m['adapter_sha256'] in allowed_adapter_hashes and pilot['adapter_sha256'] in allowed_adapter_hashes
        if v in 'ABC':
            assert not git(repo,'diff','--','*.py','*.hip','*.cpp','*.h','*.cuh'), 'Branch arithmetic modified'
    final=dict(valid=True,finished_unix=time.time(),weight_files=verified,
               source_unchanged=True,tokenizer_unchanged=True,arithmetic_unmodified_ABC=True,
               records=4320,pilot_records=540,label_schema_version=4,source_migration_verified=True)
    (ROOT/'final_audit.json').write_text(json.dumps(final,indent=2)+'\n')


def package(variants):
    out=ROOT/'packages'/variants/RESULT_PATH
    assert not out.exists(), 'Refuse to replace a previously prepared publication'
    out.mkdir(parents=True)
    for name in ('protocol_manifest.json','variant_mapping.json','source_audit.json',
                 'host_git_metadata.json','source_lock.json','checkpoint_preflight.json','final_audit.json','environment.json'):
        copy(ROOT/name,out/name)
    copy(ROOT/'protocol_zh.md',out/'protocol_zh.md')
    for name in ('measure.py','runtime_adapter.py','replay.py','summarize.py','publish_results.py','run_queue.py','test_summary.py','initialize.py','verify_label_migration.py','verify_source_migration.py','test_source_migration.py'):
        copy(ROOT/name,out/'harness'/name)
    shutil.copytree(ROOT/'naming_migration',out/'naming_migration',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'source_migration_20260926',out/'source_migration_20260926',ignore=shutil.ignore_patterns('__pycache__'))
    copy(ROOT/'prompts.jsonl',out/'prompts.jsonl')
    for stage in ('pilot','formal'):
        for v in variants:
            shutil.copytree(ROOT/'results'/stage/v,out/stage/'variants'/v)
        for name in ('summary_by_dataset.csv','per_prompt.csv','speedups.csv','report_zh.md',
                     'validation.json','cross_variant_qualification.jsonl'):
            copy(ROOT/'results'/stage/name,out/stage/name)
    # The full cross-variant record set makes every comparison independently auditable.
    for stage in ('pilot','formal'):
        copy(ROOT/'results'/stage/'runs.jsonl',out/stage/'runs.jsonl')
    for size in ('8b','14b','32b'):
        name='32b_generic_audit.json' if size=='32b' else size+'_audit.json'
        copy(WORKSPACE/'gptq_preflight_20260925'/name,out/'checkpoint_audits'/name)
    for v in variants:
        m=json.loads((ROOT/'results/formal'/v/'variant_manifest.json').read_text())
        for name in m['source_sha256']:
            if not name.endswith('.so'):
                copy(REPOS[v]/name,out/'runtime_source'/v/name)
        if v in 'DE':
            for p in (REPOS[v]/'third-party/hadacore').rglob('*'):
                if p.is_file() and p.suffix in ('.py','.hip','.h','.cpp','.cuh'):
                    copy(p,out/'runtime_source'/v/p.relative_to(REPOS[v]))
        ninja=list((REPOS[v]/'build').glob('*/build.ninja'))
        for j,p in enumerate(ninja):copy(p,out/'build_provenance'/v/f'{j}_build.ninja')
    text=f'''# Qwen3-8B A–E benchmark: {variants}

Run ID: `{RUN_ID}`. This branch owns variant(s) **{variants}**.

Read [the full report](formal/report_zh.md). A=HIP (source branch main), B=GEMM,
C=Hadacore (GEMM + HadaCore), D=Fusion, E=PARD2 (TI).
The ablation narrative is HIP -> GEMM -> Hadacore -> Fusion -> PARD2.
B adds only optimized GEMM, retaining HIP Hadamard; C retains B GEMM and
adds HadaCore. Both use the new 2026-09-26 commits, with fresh pilot runs.
Superseded pilot artifacts are archived separately with their original source identities.
Cross-variant output differences remain in all statistics. Ratios with differing
outputs describe natural-generation elapsed time, not equal-output pure speedups.

`formal/variants/` and `pilot/variants/` contain this branch's raw runs, manifests, compatibility patches,
GPU snapshots and timing/memory records. `formal/` and `pilot/` contain all five
variants' raw records and derived tables, making cross-variant comparisons auditable.
The formal workload has 96 prompts (32 per dataset), 3 repeats, natural EOS,
256 output-token cap, greedy, thinking disabled and batch size 1.
Pilot is separate: 12 disjoint prompts, cap 128.

P10/P90/P99 in primary tables use 32 per-prompt medians, linear interpolation.
Pooled-repeat percentiles are separately named. P99 is descriptive with this small
sample size. Peak allocated/reserved memory uses PyTorch counters reset per phase;
resident model memory is included. It is not whole-device VRAM usage.

Reproduction:

1. Use the recorded ROCm/PyTorch environment in `environment.json` and the exact
   source commit in `formal/variants/<variant>/variant_manifest.json`.
2. Overlay `runtime_source/<variant>/` onto that checkout. For D/E this snapshot
   preserves the actual measured local runtime independently of the result-only
   publication commit. No weights or compiled binaries are bundled.
3. Build the HIP extension using the branch's `setup.py build_ext --inplace`,
   `MAX_JOBS=2`. Build flags and binary SHA256 are recorded in build provenance
   and variant manifests. D/E also require the bundled HadaCore source build.
4. Supply the exact checkpoint, tokenizer and (for E) drafter matching the hashes
   in `protocol_manifest.json`. Do not requantize or substitute weights.
5. Set `OMP_NUM_THREADS=4`, `QUAROT_BATCHED_H128=1`,
   `QUAROT_STATIC_KV_METADATA=1`, `QUAROT_VERIFICATION_GRAPH=1`,
   `QUAROT_CHUNK_PREPROCESS=1`; run one GPU worker at a time on an idle GPU.
   Add D/E's `third-party/hadacore` directory to PYTHONPATH. Ensure Git trusts the
   explicitly selected checkout when running through a container mount.
6. Run `python harness/replay.py --variant A --repo /path/to/checkout
   --target /path/to/target --tokenizer /path/to/tokenizer
   --reference /path/to/this/folder
   --stage formal --output /path/to/new/replay_root` (one line).
   Substitute the desired variant; E additionally needs `--draft /path/to/draft`.

This run uses only GPTQ measurements. Old RTN results are separately archived;
no old C/D-labeled raw rows are mixed into this run. Current labels are A–E.
D/E use the same GPTQ target, and D supplies the paired AR oracle for E.
The PARD2 drafter remains BF16, as in the fixed test configuration.
For replay, first measure D, then reuse the same replay root for other variants.
The replay writes `results/<stage>/<variant>/` under that root and uses its
completed D records as the GPTQ oracle.

`checksums.json` covers every bundled file except itself. All validation precedes
publication. Publication adds only this result folder, without modifying kernels.
'''
    (out/'README.md').write_text(text)
    hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    assert all(p.stat().st_size<50*1024**2 for p in out.rglob('*') if p.is_file())
    (out/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return out


def main():
    assert not (ROOT/'GPTQ_REQUIRED.json').exists(), 'RTN publication disabled by the user GPTQ requirement'
    audit()
    local=WORKSPACE/RESULT_PATH
    assert not local.exists(), 'Do not replace a completed result set'
    shutil.copytree(package('ABCDE'),local)
    mapping=json.loads((ROOT/'variant_mapping.json').read_text())
    plans=[]
    for variants in ('A','B','C','DE'):
        v=variants[0];branch=mapping[v]['branch'];url=URLS['ABC' if v in 'ABC' else 'DE']
        out=package(variants)
        # Isolated checkouts preserve all existing local changes and commits.
        checkout=ROOT/'publication_checkouts'/variants
        checkout.parent.mkdir(exist_ok=True)
        subprocess.run(['git','clone','--single-branch','--branch',branch,url,str(checkout)],check=True)
        base=git(checkout,'rev-parse','HEAD')
        if v in 'ABC':
            assert base==json.loads((ROOT/'host_git_metadata.json').read_text())[v]['commit'], 'Remote source branch moved'
        shutil.copytree(out,checkout/RESULT_PATH)
        # The fused repo ignores *.json/*.log globally; explicitly include only
        # this reviewed result folder, including manifests and worker logs.
        git(checkout,'add','-f','--',str(RESULT_PATH))
        files=git(checkout,'diff','--cached','--name-only').splitlines()
        assert files and all(f.startswith(str(RESULT_PATH)+'/') for f in files)
        assert set(files)=={str(RESULT_PATH/p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
        # Preserve source snapshots byte-for-byte, including legacy whitespace.
        for relative,digest in json.loads((out/'checksums.json').read_text()).items():
            assert sha(checkout/RESULT_PATH/relative)==digest
        git(checkout,'commit','-m',f'Add validated Qwen3-8B variant {variants} benchmark results')
        plans.append((variants,checkout,branch,base,git(checkout,'rev-parse','HEAD'),url))
    receipts=[]
    for variants,checkout,branch,base,commit,url in plans:
        remote=git(checkout,'ls-remote','origin','refs/heads/'+branch).split()[0]
        assert remote==base, 'Remote moved during preparation; no automatic overwrite'
        git(checkout,'push','origin','HEAD:refs/heads/'+branch)
        assert git(checkout,'ls-remote','origin','refs/heads/'+branch).split()[0]==commit
        receipts.append(dict(variants=variants,branch=branch,commit=commit,base=base,remote=url,
                             path=str(RESULT_PATH),verified_unix=time.time()))
        (ROOT/'publication_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    copy(ROOT/'publication_receipts.json',local/'publication_receipts.json')
    registry=WORKSPACE/'benchmark_results/cumulative_qwen3_8b/index.json'
    entries=json.loads(registry.read_text())
    entries[RUN_ID].update(status='complete_and_published',report=RUN_ID+'/formal/report_zh.md')
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    index_readme=registry.parent/'README.md'
    text=index_readme.read_text().replace('A–E，A 正式執行中；新版 B/C 已替換，待重測及全輪驗證','A–E，已完成驗證與遠端推送')
    text=text.replace('完成後建立 `qwen3_8b_abcde_gptq_20260925_01/`',
        '[正式報告](qwen3_8b_abcde_gptq_20260925_01/formal/report_zh.md)')
    text=text.replace('A 結束後先測新版 B/C pilot，再測 B/C/E formal。','新版 B/C pilot 與完整 A–E formal 已完成驗證。')
    index_readme.write_text(text)
    print('Published and remote-verified all four result commits.',flush=True)


if __name__=='__main__':
    main()
