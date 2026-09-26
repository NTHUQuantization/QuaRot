"""Maintainer-only CPU tool, already executed for the distributed bundle.

Do not regenerate hashes on the measurement host to bypass a mismatch.
"""
import argparse
import json
from pathlib import Path
from common import ROOT, load_prompts, ids_sha256, sha256


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads((ROOT / 'protocol.json').read_text())
    model_hashes = json.loads((ROOT / 'model_hashes.json').read_text())
    lengths = {}
    for size, model in spec['models'].items():
        snapshot = args.cache_root / ('models--' + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']
        for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'generation_config.json']:
            assert sha256(snapshot / name) == model_hashes[size][name], name
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        assert tokenizer.convert_tokens_to_ids('</think>') == 151668
        config = json.loads((snapshot / 'config.json').read_text())
        lengths[size] = {}
        for task in spec['tasks']:
            rows = []
            for row in load_prompts(task):
                ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=True)
                assert tokenizer.decode(ids[-4:], skip_special_tokens=False).endswith('<|im_start|>assistant\n')
                assert len(ids) + spec['generation']['max_new_tokens'] + 15 <= config['max_position_embeddings']
                rows.append({'id': row['id'], 'input_tokens': len(ids), 'input_ids_sha256': ids_sha256(ids)})
            dest = ROOT / 'input_contracts' / f'{size}_{task}.jsonl'
            data = ''.join(json.dumps(row) + '\n' for row in rows)
            if dest.exists() and dest.read_text() != data:
                raise RuntimeError(f'Existing input contract changed: {dest}')
            dest.write_text(data)
            lengths[size][task] = {'count': len(rows), 'max_input_tokens': max(r['input_tokens'] for r in rows)}
    (ROOT / 'input_contracts' / 'validation.json').write_text(json.dumps(lengths, indent=2) + '\n')
    print(json.dumps(lengths, indent=2))


if __name__ == '__main__':
    main()
