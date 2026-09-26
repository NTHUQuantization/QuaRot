"""Prove that the adapter change only permuted the B/C source assignments."""
import ast
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    evidence=root/'naming_migration'
    metadata=json.loads((evidence/'manifest.json').read_text())
    old=evidence/'runtime_adapter_label_v2.py'
    assert digest(old)==metadata['old_adapter_sha256']
    assert digest(root/'measure.py')==metadata['old_worker_sha256'], 'Timed measurement worker changed'
    before=ast.parse(old.read_text())
    after=ast.parse((root/'runtime_adapter.py').read_text())
    assignments=[n for n in before.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='BRANCHES' for t in n.targets)]
    assert len(assignments)==1 and isinstance(assignments[0].value,ast.Dict)
    mapping=assignments[0].value
    keys=[key.value for key in mapping.keys]
    b,c=keys.index('B'),keys.index('C')
    mapping.values[b],mapping.values[c]=mapping.values[c],mapping.values[b]
    assert ast.dump(before)==ast.dump(after), 'Adapter changed beyond the B/C naming permutation'
    assert digest(evidence/'original_label_v2_records.tar.gz')==metadata['archive_sha256']
    return {metadata['old_adapter_sha256'],digest(root/'runtime_adapter.py')}


if __name__=='__main__':
    from pathlib import Path
    print('Verified labeling-only adapter change:',sorted(verify(Path(__file__).resolve().parent)))
