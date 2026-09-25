"""Audit and summarize one independently measured implementation."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

DATASETS=('humaneval','gsm8k','math_500')
PHASES=('prefill','decode','e2e')


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def quantile(values,q):
    values=sorted(values)
    x=(len(values)-1)*q
    lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def audit(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    done=json.loads((root/'complete.json').read_text())
    stage=manifest['stage'];label=manifest['label']
    assert stage in ('pilot','formal')
    count=12 if stage=='pilot' else 96
    cap=128 if stage=='pilot' else 256
    prompts={(p['dataset'],p['sample_id']):p for p in read(root/'prompts.jsonl') if p['stage']==stage}
    assert len(prompts)==count
    assert done['stage']==stage and done['label']==label and done['records']==count*9 and done['all_within_run_exact']
    assert manifest['cap']==cap and manifest['repeats']==3
    assert hashlib.sha256((root/'prompts.jsonl').read_bytes()).hexdigest()==manifest['prompt_file_sha256']
    contract=json.loads((root/'contract.json').read_text())
    assert contract==manifest['contract']
    from run import identity
    assert contract['comparison_id']==identity(contract['common'])
    assert contract['common']['prompts_sha256']==manifest['prompt_file_sha256']
    assert contract['common']['profile_sha256']==hashlib.sha256((root/'profile.json').read_bytes()).hexdigest()
    assert contract['common']['worker_sha256']==manifest['worker_sha256']
    assert contract['adapter_sha256']==manifest['adapter_sha256']
    assert contract['common']['target_sha256']==contract['inputs']['target']
    assert contract['common']['tokenizer_sha256']==contract['inputs']['tokenizer']
    rows=read(root/'runs.jsonl')
    expected={(d,i,s,p) for d,i in prompts for s in range(3) for p in PHASES}
    assert len(rows)==len(expected)
    assert {(r['dataset'],r['sample_id'],r['sweep'],r['phase']) for r in rows}==expected
    groups={};signatures={}
    for r in rows:
        key=r['dataset'],r['sample_id']
        assert r['label']==label and r['stage']==stage
        assert r['run_id']==contract['config']['run_id']
        assert r['valid'] and not r['gpu_competition'] and math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>0
        assert r['input_hash']==prompts[key]['input_hash'] and r['input_tokens']==len(prompts[key]['input_ids'])
        assert r['peak_reserved_bytes']>=r['peak_allocated_bytes']>=r['baseline_allocated_bytes']>0
        assert r['incremental_peak_allocated_bytes']==r['peak_allocated_bytes']-r['baseline_allocated_bytes']
        groups.setdefault((*key,r['phase']),[]).append(r)
        if r['phase']!='prefill':
            ids=r['output_ids']
            assert 0<len(ids)<=cap and len(ids)==r['output_tokens']
            assert r['output_hash']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert not any(t in manifest['eos_ids'] for t in ids[:-1])
            assert r['stop_reason']==('eos' if ids[-1] in manifest['eos_ids'] else 'cap')
            assert r['cap_reached']==(len(ids)==cap)
            assert r['stop_reason']=='eos' or len(ids)==cap
            if 'emitted_tokens_per_step' in r['raw_stats']:
                assert sum(r['raw_stats']['emitted_tokens_per_step'])==len(ids)
            assert signatures.setdefault(key,ids)==ids,'Within-run output mismatch'
    qualification=read(root/'qualification.jsonl')
    assert len(qualification)==count*6 and all(r['within_run_exact'] for r in qualification)
    assert {(r['dataset'],r['sample_id'],r['sweep'],r['phase']) for r in qualification}=={
        (d,i,s,p) for d,i in prompts for s in range(3) for p in ('decode','e2e')}
    gpu=read(root/'gpu.jsonl')
    assert len(gpu)==count*3 and all(r['returncode']==0 and len(r['gpu_processes'])==1 for r in gpu)
    per_prompt=[]
    for (d,i,phase),items in groups.items():
        assert len(items)==3
        n=len(signatures[d,i])
        representative=next(r for r in rows if (r['dataset'],r['sample_id'],r['phase'])==(d,i,'e2e'))
        per_prompt.append(dict(label=label,dataset=d,sample_id=i,phase=phase,
            input_tokens=prompts[d,i]['input_tokens'],output_tokens=n,
            stop_reason=representative['stop_reason'],cap_reached=representative['cap_reached'],
            median_ms=st.median(r['elapsed_ms'] for r in items),
            median_output_tokens_per_second=st.median(n*1000/r['elapsed_ms'] for r in items) if phase!='prefill' else '',
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in items),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in items)))
    return manifest,rows,per_prompt,signatures


def main(root):
    root=Path(root)
    # A previously generated success marker must not survive a failed re-audit.
    (root/'validation.json').unlink(missing_ok=True)
    manifest,rows,per_prompt,signatures=audit(root)
    summaries=[]
    for dataset,phase in itertools.product(DATASETS,PHASES):
        raw=[r for r in rows if (r['dataset'],r['phase'])==(dataset,phase)]
        pp=[r for r in per_prompt if (r['dataset'],r['phase'])==(dataset,phase)]
        medians=[r['median_ms'] for r in pp];pooled=[r['elapsed_ms'] for r in raw]
        summaries.append(dict(label=manifest['label'],dataset=dataset,phase=phase,prompts=len(pp),repeats=3,
            median_ms=st.median(medians),p10_ms=quantile(medians,.1),p90_ms=quantile(medians,.9),p99_ms=quantile(medians,.99),
            pooled_p10_ms=quantile(pooled,.1),pooled_p90_ms=quantile(pooled,.9),pooled_p99_ms=quantile(pooled,.99),
            median_output_tokens_per_second=st.median(r['median_output_tokens_per_second'] for r in pp) if phase!='prefill' else '',
            output_tokens_median=st.median(r['output_tokens'] for r in pp),
            cap_reached_fraction=st.mean(r['cap_reached'] for r in pp),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in raw),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in raw),
            peak_incremental_allocated_bytes=max(r['incremental_peak_allocated_bytes'] for r in raw)))
    csv_write(root/'per_prompt.csv',per_prompt)
    csv_write(root/'summary_by_dataset.csv',summaries)
    report=f"# {manifest['label']} — {manifest['stage']}\n\n"
    report+='每題三次 median 後取跨題 median；p10/p90/p99 為每題 median 分布的線性插值。Peak 為 PyTorch allocated/reserved，含常駐模型；非整卡 VRAM。\n\n'
    report+='|Dataset|Phase|Median ms|p10 ms|p90 ms|p99 ms|tok/s|Peak allocated GiB|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summaries:
        rate=r['median_output_tokens_per_second']
        rate=f'{rate:.2f}' if rate!='' else '—'
        report+=f"|{r['dataset']}|{r['phase']}|{r['median_ms']:.2f}|{r['p10_ms']:.2f}|{r['p90_ms']:.2f}|{r['p99_ms']:.2f}|{rate}|{r['peak_allocated_bytes']/2**30:.3f}|\n"
    report+='\n同版重複及 decode/E2E 輸出逐 token 一致。跨實作比較請用 compare.py；p99 樣本少，只作描述，不代表穩定服務尾延遲。\n'
    (root/'report_zh.md').write_text(report)
    result=dict(valid=True,label=manifest['label'],stage=manifest['stage'],records=len(rows),prompts=len(signatures),within_run_exact=True)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    main(p.parse_args().run)
