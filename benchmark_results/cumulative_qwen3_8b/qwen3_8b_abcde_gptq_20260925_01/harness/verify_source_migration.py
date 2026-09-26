"""Verify pinned source revisions and prove the timing/adapter computation is unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, variants='ABCDE'):
    from verify_label_migration import verify as verify_prior_labels
    from runtime_adapter import BRANCHES
    from measure import snapshot
    allowed=verify_prior_labels(root)
    evidence=root/'source_migration_20260926'
    metadata=json.loads((evidence/'manifest.json').read_text())
    before_file=evidence/'previous_schema3/runtime_adapter.py'
    assert digest(before_file)==metadata['adapter_before_sha256']
    assert digest(root/'measure.py')==metadata['old_worker_sha256'], 'Timing worker changed'
    assert digest(root/'runtime_adapter.py')==metadata['adapter_after_sha256']
    before=ast.parse(before_file.read_text())
    assignments=[n for n in before.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='BRANCHES' for t in n.targets)]
    assert len(assignments)==1
    mapping=assignments[0].value
    keys=[key.value for key in mapping.keys]
    for v in 'BC':
        expr="ROOT / "+repr(metadata['new_sources'][v]['relative_checkout'])
        mapping.values[keys.index(v)]=ast.parse(expr,mode='eval').body
    assert ast.dump(before)==ast.dump(ast.parse((root/'runtime_adapter.py').read_text())), 'Adapter computation changed'
    assert digest(evidence/'superseded_gptq_pilot_schema3.tar.gz')==metadata['archive_sha256']
    old=json.loads((evidence/'previous_schema3/protocol_manifest.json').read_text())
    current=json.loads((root/'protocol_manifest.json').read_text())
    metadata_only={'variants','source_audit','ordering','label_schema_version','protocol_sha256','source_migration'}
    for name in set(old)-metadata_only:
        assert current[name]==old[name], f'Common workload/weights/settings changed: {name}'
    lock=json.loads((root/'source_lock.json').read_text())
    host=json.loads((root/'host_git_metadata.json').read_text())
    names=json.loads((root/'variant_mapping.json').read_text())
    for v in variants:
        repo=BRANCHES[v]
        record=lock['variants'][v]
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==record['commit']==host[v]['commit']
        assert snapshot(repo)==record['source_sha256'], f'Pinned source/binary changed: {v}'
        if v in 'BC':
            assert record['commit']==metadata['new_sources'][v]['commit']==names[v]['commit']
            assert names[v]['branch']==metadata['new_sources'][v]['branch']
            assert repo.resolve()==(root/metadata['new_sources'][v]['relative_checkout']).resolve()
            build=json.loads((evidence/f'build_{v}.json').read_text())
            extension=list((repo/'quarot').glob('_HIP*.so'))
            assert len(extension)==1 and digest(extension[0])==build['extension_sha256']
            assert build['status']=='built' and build['commit']==record['commit']
        if v in 'ABC':
            diff=subprocess.check_output(['git','-C',str(repo),'diff','--','*.py','*.hip','*.cpp','*.h','*.cuh'],text=True)
            assert not diff, f'Branch arithmetic modified: {v}'
    return allowed|{metadata['adapter_after_sha256']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list('ABCDE'))
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    print('Verified source schema 4; B=GEMM 973a13d, C=Hadacore 6c30374; worker and adapter computation unchanged:',
          sorted(verify(root,args.variant or 'ABCDE')),flush=True)
