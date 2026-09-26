"""Validate complete A-E records and rebuild all tables from raw measurements."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st

ROOT = Path(__file__).resolve().parent
DATASETS = ('humaneval', 'gsm8k', 'math_500')
PHASES = ('prefill', 'decode', 'e2e')
NAMES = dict(A='HIP', B='Hadacore', C='GEMM', D='Fusion', E='PARD2')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def quantile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    lo = int(x)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(stage='formal'):
    root = ROOT/'results'/stage
    prompts = { (p['dataset'],p['sample_id']): p for p in
               read(ROOT/'prompts.jsonl') if p['stage'] == stage }
    count = 96 if stage == 'formal' else 12
    cap = 256 if stage == 'formal' else 128
    assert len(prompts) == count
    all_rows, per_prompt, summaries, signatures, manifests = [], [], [], {}, {}
    for v in 'ABCDE':
        folder = root/v
        done = json.loads((folder/'complete.json').read_text())
        assert done['records'] == count*9 and done['all_within_variant_exact']
        m = manifests[v] = json.loads((folder/'variant_manifest.json').read_text())
        assert m['variant'] == v and m['stage'] == stage and m['cap'] == cap
        records = read(folder/'runs.jsonl')
        expected = {(d,i,s,p) for d,i in prompts for s in range(3) for p in PHASES}
        assert len(records) == len(expected)
        assert {(r['dataset'],r['sample_id'],r['sweep'],r['phase']) for r in records} == expected
        groups = {}
        for r in records:
            k = r['dataset'],r['sample_id']
            assert r['variant'] == v and r['stage'] == stage
            assert r['valid'] and not r['gpu_competition'] and math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0
            assert r['input_hash'] == prompts[k]['input_hash'] and r['input_tokens'] == len(prompts[k]['input_ids'])
            assert r['peak_reserved_bytes'] >= r['peak_allocated_bytes'] >= r['baseline_allocated_bytes'] > 0
            assert r['incremental_peak_allocated_bytes'] == r['peak_allocated_bytes']-r['baseline_allocated_bytes']
            groups.setdefault((*k,r['phase']),[]).append(r)
            if r['phase'] != 'prefill':
                ids = r['output_ids']
                assert r['output_tokens'] == len(ids) and 0 < len(ids) <= cap
                assert r['output_hash'] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                assert not any(t in m['eos_ids'] for t in ids[:-1])
                assert r['stop_reason'] == ('eos' if ids[-1] in m['eos_ids'] else 'cap')
                assert r['cap_reached'] == (len(ids) == cap)
                assert r['stop_reason'] == 'eos' or len(ids) == cap
                assert sum(r['raw_stats']['emitted_tokens_per_step']) == len(ids)
                signature = (ids,r['stop_reason'],r['cap_reached'])
                assert signatures.setdefault((v,*k),signature) == signature
        q = read(folder/'qualification.jsonl')
        assert len(q) == count*6 and all(r['within_variant_exact'] for r in q)
        gpu = read(folder/'gpu.jsonl')
        assert len(gpu) == count*3 and all(r['returncode'] == 0 and len(r['gpu_processes']) == 1 for r in gpu)
        for (d,i,p), rows in groups.items():
            assert len(rows) == 3
            n = len(signatures[v,d,i][0])
            per_prompt.append(dict(variant=v,variant_name=NAMES[v],dataset=d,sample_id=i,phase=p,
                input_tokens=prompts[d,i]['input_tokens'],output_tokens=n,
                stop_reason=signatures[v,d,i][1],cap_reached=signatures[v,d,i][2],
                median_ms=st.median(r['elapsed_ms'] for r in rows),
                median_output_tokens_per_second=st.median(n*1000/r['elapsed_ms'] for r in rows) if p!='prefill' else '',
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows)))
        for d,p in itertools.product(DATASETS,PHASES):
            rows = [r for r in records if r['dataset']==d and r['phase']==p]
            pp = [r for r in per_prompt if r['variant']==v and r['dataset']==d and r['phase']==p]
            medians = [r['median_ms'] for r in pp]
            pooled = [r['elapsed_ms'] for r in rows]
            summaries.append(dict(variant=v,variant_name=NAMES[v],dataset=d,phase=p,prompts=len(pp),repeats=3,
                median_ms=st.median(medians),
                p10_ms=quantile(medians,.1),p90_ms=quantile(medians,.9),p99_ms=quantile(medians,.99),
                pooled_p10_ms=quantile(pooled,.1),pooled_p90_ms=quantile(pooled,.9),pooled_p99_ms=quantile(pooled,.99),
                median_output_tokens_per_second=st.median(r['median_output_tokens_per_second'] for r in pp) if p!='prefill' else '',
                input_tokens_min=min(r['input_tokens'] for r in pp),input_tokens_median=st.median(r['input_tokens'] for r in pp),input_tokens_max=max(r['input_tokens'] for r in pp),
                output_tokens_min=min(r['output_tokens'] for r in pp),output_tokens_median=st.median(r['output_tokens'] for r in pp),output_tokens_max=max(r['output_tokens'] for r in pp),
                cap_reached_fraction=st.mean(r['cap_reached'] for r in pp),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
                peak_incremental_allocated_bytes=max(r['incremental_peak_allocated_bytes'] for r in rows)))
        all_rows += records
    assert len({tuple(m['eos_ids']) for m in manifests.values()}) == 1
    assert len({m['prompt_file_sha256'] for m in manifests.values()}) == 1
    assert manifests['D']['source_sha256'] == manifests['E']['source_sha256']
    flags = [dict(manifests[v]['runtime_provenance']['runtime_flags']) for v in 'DE']
    for f in flags:
        f.pop('execution_policy',None)
    assert flags[0] == flags[1], 'D/E target execution flags differ'
    cross=[]
    for a,b in itertools.combinations('ABCDE',2):
        for d,i in prompts:
            x,y = signatures[a,d,i][0],signatures[b,d,i][0]
            cross.append(dict(left=a,right=b,dataset=d,sample_id=i,exact=x==y,
                left_tokens=len(x),right_tokens=len(y),
                first_difference=next((j for j,(xx,yy) in enumerate(zip(x,y)) if xx!=yy), min(len(x),len(y)) if len(x)!=len(y) else None)))
    assert all(r['exact'] for r in cross if (r['left'],r['right'])==('D','E')), 'PARD2 disagrees with paired AR'
    times = {(r['variant'],r['dataset'],r['sample_id']):r['median_ms'] for r in per_prompt if r['phase']=='e2e'}
    ratios=[]
    pairs = [('A',v) for v in 'BCDE'] + [('B','C'),('C','D'),('D','E')]
    for a,b in pairs:
        logs={d:[math.log(times[a,dd,i]/times[b,dd,i]) for dd,i in prompts if dd==d] for d in DATASETS}
        for dataset in (*DATASETS,'dataset_equal_overall'):
            use = DATASETS if dataset=='dataset_equal_overall' else (dataset,)
            estimate = math.exp(st.mean(st.mean(logs[d]) for d in use))
            rng=random.Random(0)
            boots=[math.exp(st.mean(st.mean(rng.choices(logs[d],k=len(logs[d]))) for d in use)) for _ in range(2000)]
            match=[r for r in cross if (r['left'],r['right'])==(a,b) and r['dataset'] in use]
            ratios.append(dict(left=a,left_name=NAMES[a],right=b,right_name=NAMES[b],dataset=dataset,paired_geomean_e2e_time_ratio=estimate,
                bootstrap_ci95_low=quantile(boots,.025),bootstrap_ci95_high=quantile(boots,.975),
                identical_output_prompts=sum(r['exact'] for r in match),prompts=len(match),
                interpretation='equal-output speedup' if all(r['exact'] for r in match) else 'natural-generation time ratio; output workloads differ'))
    for name,rows in [('per_prompt.csv',per_prompt),('summary_by_dataset.csv',summaries),('speedups.csv',ratios)]:
        csv_write(root/name,rows)
    for name,rows in [('runs.jsonl',all_rows),('cross_variant_qualification.jsonl',cross)]:
        (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report = f'# Qwen3-8B GPTQ A–E 指定分支比較：{stage}\n\n'
    report += f'{count} 題；每資料集 {count//3} 題；batch=1；greedy；thinking=false；自然 EOS；新增 token 上限 {cap}；每題 3 sweeps。E=PARD2-TI，draft_k=15。\n\n'
    report += 'A=HIP（來源 branch main）；B=Hadacore；C=GEMM；D=Fusion；E=PARD2（TI）。五組共用 GPTQ W4A4KV4 checkpoint，grouped_h256_v1、activation clip=0.9。消融敘事順序：HIP → Hadacore → GEMM → Fusion → PARD2。B 保留 HIP 的 GEMM 並加入 HadaCore；較新的 C 保留 HadaCore 並加入 GEMM 優化。各組保留來源分支 kernel，僅適配 checkpoint 欄位與 cache API。\n\n'
    report += '時間與 tokens/s 先取每題 3 次 median，再取跨題 median。P10/P90/P99 以每題 median 分布、線性插值計算；CSV 另列 pooled repeats 分布。P99 樣本少，只作描述，不能解讀為穩定服務尾延遲。\n\n'
    report += 'Peak memory 是 phase reset 後 PyTorch allocated／reserved 峰值，含常駐模型；不是整卡 VRAM。表列該資料集 E2E 最大 allocated 峰值，其他 phases 與 reserved 見 CSV。E2E 不含模型載入、tokenization 或網路。\n\n'
    for d in DATASETS:
        report += f'## {d}\n\n|版本|Prefill ms|Decode ms|E2E ms|E2E p10 / p90 ms|Decode tok/s|E2E tok/s|Output tokens median|Cap 比例|Peak allocated GiB|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for v in 'ABCDE':
            vals={r['phase']:r for r in summaries if r['variant']==v and r['dataset']==d}
            p,dec,e=vals['prefill'],vals['decode'],vals['e2e']
            report += f"|{v} {NAMES[v]}|{p['median_ms']:.2f}|{dec['median_ms']:.2f}|{e['median_ms']:.2f}|{e['p10_ms']:.2f} / {e['p90_ms']:.2f}|{dec['median_output_tokens_per_second']:.2f}|{e['median_output_tokens_per_second']:.2f}|{e['output_tokens_median']:g}|{e['cap_reached_fraction']:.1%}|{e['peak_allocated_bytes']/2**30:.3f}|\n"
        report += '\n'
    report += '## 配對 E2E 耗時比（資料集等權）\n\n分子為左版時間、分母為右版時間。輸出不同時，數值包含生成軌跡／長度差異，不宣稱等輸出純加速。\n\n|比較|耗時比|95% bootstrap CI|等輸出題數|\n|---|---:|---:|---:|\n'
    for r in ratios:
        if r['dataset']=='dataset_equal_overall':
            report += f"|{r['left']}→{r['right']}|{r['paired_geomean_e2e_time_ratio']:.3f}×|{r['bootstrap_ci95_low']:.3f}–{r['bootstrap_ci95_high']:.3f}|{r['identical_output_prompts']}/{r['prompts']}|\n"
    report += '\n全部版本同版跨 repeats／decode／E2E 輸出一致；D／E 逐題等輸出。跨版本差異與首次差異 token 位置見 cross_variant_qualification.jsonl；未刪除差異題。\n\n'
    report += f'各版本依序測量，未鎖定 GPU clocks；時間順序可能帶來溫度／時脈漂移。GPU 快照與實際執行成本保存在各版本資料夾。{cap}-token 上限可能截斷答案；本報告是生成效能測試，不代表任務答對率。\n'
    (root/'report_zh.md').write_text(report)
    audit=dict(valid=True,stage=stage,prompts=count,records=len(all_rows),within_variant_exact=True,
        D_E_exact=True,cross_pairs={a+b:sum(r['exact'] for r in cross if r['left']==a and r['right']==b) for a,b in itertools.combinations('ABCDE',2)},
        complete_variants=list('ABCDE'),memory_recorded=True)
    (root/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'formal')
