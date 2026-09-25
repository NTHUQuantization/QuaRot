"""Compare arbitrary named runs after checking workload, environment and outputs."""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st
from summarize import audit, csv_write, quantile, DATASETS


def compare(paths, allow_environment_difference=False):
    runs=[audit(p) for p in paths]
    if len(runs)<2:raise ValueError('Supply at least two complete runs')
    manifests=[r[0] for r in runs]
    labels=[m['label'] for m in manifests]
    if len(set(labels))!=len(labels):raise ValueError('Use distinct implementation labels')
    if len({m['stage'] for m in manifests})!=1:raise ValueError('Cannot mix pilot and formal')
    if len({m['contract']['comparison_id'] for m in manifests})!=1:
        raise ValueError('Incompatible workload/weights/timing contract')
    if len({tuple(m['eos_ids']) for m in manifests})!=1:raise ValueError('EOS definitions differ')
    env_keys=('torch','hip','python','gpu','dependencies','host_cpu')
    environments=[{k:m.get(k) for k in env_keys} for m in manifests]
    environment_matches=all(e==environments[0] for e in environments)
    if not environment_matches and not allow_environment_difference:
        raise ValueError('Environment differs; inspect manifests or explicitly allow an environment comparison')
    rows=[];outputs=[]
    for left,right in itertools.combinations(range(len(runs)),2):
        lm,_,lp,ls=runs[left];rm,_,rp,rs=runs[right]
        assert set(ls)==set(rs)
        for key,ids in ls.items():
            other=rs[key]
            outputs.append(dict(left=lm['label'],right=rm['label'],dataset=key[0],sample_id=key[1],
                exact=ids==other,left_tokens=len(ids),right_tokens=len(other),
                first_difference=next((i for i,(a,b) in enumerate(zip(ids,other)) if a!=b),
                    min(len(ids),len(other)) if len(ids)!=len(other) else None)))
        lt={(p['dataset'],p['sample_id']):p['median_ms'] for p in lp if p['phase']=='e2e'}
        rt={(p['dataset'],p['sample_id']):p['median_ms'] for p in rp if p['phase']=='e2e'}
        for dataset in (*DATASETS,'dataset_equal_overall'):
            use=DATASETS if dataset=='dataset_equal_overall' else (dataset,)
            logs={d:[math.log(lt[k]/rt[k]) for k in lt if k[0]==d] for d in use}
            rng=random.Random(0)
            boots=[math.exp(st.mean(st.mean(rng.choices(logs[d],k=len(logs[d]))) for d in use)) for _ in range(2000)]
            exact=sum(ls[k]==rs[k] for k in ls if k[0] in use)
            count=sum(k[0] in use for k in ls)
            interpretation='equal-output speedup' if exact==count else 'natural-generation time ratio; output workloads differ'
            if not environment_matches:interpretation='environment differs; '+interpretation
            rows.append(dict(left=lm['label'],right=rm['label'],dataset=dataset,
                paired_geomean_e2e_time_ratio=math.exp(st.mean(st.mean(logs[d]) for d in use)),
                bootstrap_ci95_low=quantile(boots,.025),bootstrap_ci95_high=quantile(boots,.975),
                identical_output_prompts=exact,prompts=count,environment_matches=environment_matches,
                interpretation=interpretation))
    return rows,outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-environment-difference',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Choose a new comparison output folder')
    rows,outputs=compare(args.runs,args.allow_environment_difference)
    args.output.mkdir(parents=True)
    csv_write(args.output/'time_ratios.csv',rows)
    (args.output/'output_comparison.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    (args.output/'comparison.json').write_text(json.dumps(dict(runs=[str(p.resolve()) for p in args.runs],
        allow_environment_difference=args.allow_environment_difference),indent=2)+'\n')
    report='# Benchmark comparison\n\n分子為左版時間，分母為右版；依 prompt 配對、資料集等權。\n\n'
    report+='|Left|Right|E2E ratio|95% CI|Exact outputs|Interpretation|\n|---|---|---:|---|---|---|\n'
    for r in rows:
        if r['dataset']=='dataset_equal_overall':
            report+=f"|{r['left']}|{r['right']}|{r['paired_geomean_e2e_time_ratio']:.3f}|{r['bootstrap_ci95_low']:.3f}–{r['bootstrap_ci95_high']:.3f}|{r['identical_output_prompts']}/{r['prompts']}|{r['interpretation']}|\n"
    (args.output/'report_zh.md').write_text(report)


if __name__=='__main__':main()
