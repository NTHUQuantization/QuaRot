"""Freeze the new GPTQ run only after all requested checkpoints pass audit."""
import json
from pathlib import Path
import subprocess
import time

from runtime_adapter import ROOT,WORKSPACE,BRANCHES,TARGET,DRAFT
from measure import sha


def main():
    preparation=WORKSPACE/'gptq_preflight_20260925'
    while True:
        state=json.loads((preparation/'status.json').read_text())
        if state['status']=='complete':break
        assert state['status']=='running',state
        time.sleep(15)
    if (ROOT/'initialized.json').exists():return
    audits={}
    for size in ('8b','14b','32b'):
        filename='32b_generic_audit.json' if size=='32b' else size+'_audit.json'
        audit=json.loads((preparation/filename).read_text())
        assert audit['status']=='passed' and audit['method']=='gptq'
        audits[size]=audit
    current=json.loads((TARGET/'config.json').read_text())
    assert current['quarot_conversion']==audits['8b']['conversion']
    common=json.loads((WORKSPACE/'cumulative_ablation_abcde_20260925/protocol_manifest.json').read_text())
    common.update(run_id=ROOT.name,scope=list('ABCDE'),created_unix=time.time(),
        target='/workspace_root/'+TARGET.name,quantization_method='gptq',
        source_audit=json.loads((ROOT/'source_audit.json').read_text()),
        ffn_rotation='grouped_h256_v1',activation_clip_ratio=.9,
        ordering='Each stage D then A/B/C/E on one GPU; D establishes the GPTQ AR oracle',
        protocol_sha256=sha(ROOT/'protocol_zh.md'),
        historical_comparison='RTN results are separately archived; no RTN timing/output data enter this GPTQ report',
        checkpoint_audits={size:dict(path=audit['checkpoint'],method='gptq',sha256=audit['sha256']) for size,audit in audits.items()})
    weights={'/workspace_root/'+TARGET.name+'/'+name:digest for name,digest in audits['8b']['sha256'].items()}
    for path in DRAFT.iterdir():
        if path.name in ('config.json','model.safetensors','warp_model.bin'):
            weights['/workspace_root/'+str(path.relative_to(WORKSPACE))]=sha(path)
    common['weight_files']=weights
    for name,digest in audits['8b']['sha256'].items():assert sha(TARGET/name)==digest
    metadata={}
    for v,repo in BRANCHES.items():
        git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        metadata[v]=dict(commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
                         status=git('status','--short'),remote=git('remote','get-url','origin'))
    for name,obj in [('protocol_manifest.json',common),('host_git_metadata.json',metadata),
                     ('checkpoint_preflight.json',dict(verified_unix=time.time(),sha256=weights,method='gptq')),
                     ('initialized.json',dict(complete=True,initialized_unix=time.time()))]:
        (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print('GPTQ run initialized; all three model sizes verified.',flush=True)
    registry=WORKSPACE/'benchmark_results/cumulative_qwen3_8b/index.json'
    entries=json.loads(registry.read_text())
    entries[ROOT.name]['status']='measurement_running'
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
