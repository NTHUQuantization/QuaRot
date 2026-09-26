"""CPU-only contract helpers; generation must never read ground_truth/."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_bundle():
    expected = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    for name, digest in expected.items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f'Changed contract file: {name}')
    return sha256(ROOT / 'SHA256SUMS.json')


def sample_seed(task, item_id, replicate_seed):
    text = f'qwen3-thinking-quality-v2-12h|{task}|{item_id}|{replicate_seed}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big') % (2**31 - 1)


def ids_sha256(ids):
    """Canonical UTF-8 JSON list of Python ints, without whitespace."""
    return hashlib.sha256(json.dumps([int(x) for x in ids], separators=(',', ':')).encode()).hexdigest()


def final_ids(output_ids, think_end_id=151668):
    """Only generated tokens are passed in; do not rely on an opening tag.

    Missing or repeated closing markers are invalid protocol outputs, never
    candidates for answer extraction. A capped output may still have a final.
    """
    positions = [i for i, token in enumerate(output_ids) if token == think_end_id]
    if len(positions) != 1:
        return [], {'thinking_complete': False, 'think_end_count': len(positions),
                    'reason': 'missing_think_end' if not positions else 'multiple_think_end'}
    i = positions[0]
    return output_ids[i + 1:], {'thinking_complete': True, 'think_end_count': 1,
                               'thinking_tokens_including_end': i + 1}


def load_prompts(task):
    return [json.loads(line) for line in (ROOT / 'prompts' / f'{task}.jsonl').read_text().splitlines()]


def verify_tokenized(tokenizer, size, task):
    expected = {r['id']: r for r in
                [json.loads(line) for line in (ROOT / 'input_contracts' / f'{size}_{task}.jsonl').read_text().splitlines()]}
    count = 0
    for row in load_prompts(task):
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=True)
        wanted = expected[row['id']]
        if len(ids) != wanted['input_tokens'] or ids_sha256(ids) != wanted['input_ids_sha256']:
            raise RuntimeError(f'Tokenizer/prompt mismatch: {size}/{task}/{row["id"]}')
        count += 1
    if count != len(expected):
        raise RuntimeError('Incomplete input contract')
    return count
