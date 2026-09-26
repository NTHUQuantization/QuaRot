"""No GPU/model imports: verify scoring boundaries that can change accuracy."""
import json
import unittest
from common import ROOT, final_ids, sample_seed, load_prompts
from scoring import score_math_final, code_candidate


class ContractTests(unittest.TestCase):
    def test_incomplete_thinking_has_no_final(self):
        result, meta = final_ids([12, 34, 56])
        self.assertEqual(result, [])
        self.assertFalse(meta['thinking_complete'])

    def test_template_opening_is_not_required_in_output(self):
        result, meta = final_ids([12, 151668, 42, 151645])
        self.assertEqual(result, [42, 151645])
        self.assertTrue(meta['thinking_complete'])

    def test_repeated_end_rejected(self):
        self.assertFalse(final_ids([151668, 42, 151668])[1]['thinking_complete'])

    def test_numeric_exact_and_no_reasoning_fallback(self):
        self.assertTrue(score_math_final('gsm8k', r'Answer: \boxed{\frac{1}{2}}', '0.5')['correct'])
        self.assertTrue(score_math_final('gsm8k', r'\boxed{1,200}', '1200')['correct'])
        self.assertFalse(score_math_final('gsm8k', 'I calculated 1200', '1200')['correct'])
        self.assertFalse(score_math_final('gsm8k', r'\boxed{1200', '1200')['correct'])
        self.assertFalse(score_math_final('gsm8k', r'\boxed{x = 1200}', '1200')['correct'])

    def test_math_normalization_is_conservative(self):
        self.assertTrue(score_math_final('math_500', r'\boxed{\left(3,\frac{\pi}{2}\right)}', r'(3,\frac{\pi}{2})')['correct'])
        self.assertFalse(score_math_final('math_500', r'\boxed{x+x}', '2x')['correct'])

    def test_code_imports_retained(self):
        answer = '```python\nimport math\ndef f(x):\n    return math.sqrt(x)\n```'
        self.assertTrue(code_candidate('def f(x):\n    pass', answer, 'f').startswith('import math'))

    def test_prompts_have_no_truth_fields(self):
        contract = json.loads((ROOT / 'protocol.json').read_text())
        for task, spec in contract['tasks'].items():
            rows = load_prompts(task)
            self.assertEqual(len(rows), spec['count'])
            self.assertEqual(len({r['id'] for r in rows}), spec['count'])
            for row in rows:
                self.assertEqual(set(row), {'id', 'messages'})
                self.assertEqual([r['role'] for r in row['messages']], ['system', 'user'])

    def test_seed_depends_on_sample_not_execution_order(self):
        a = sample_seed('gsm8k', 'gsm8k/test/0', 20260926)
        self.assertEqual(a, sample_seed('gsm8k', 'gsm8k/test/0', 20260926))
        self.assertNotEqual(a, sample_seed('gsm8k', 'gsm8k/test/1', 20260926))
        self.assertNotEqual(a, sample_seed('gsm8k', 'gsm8k/test/0', 20260927))


if __name__ == '__main__':
    unittest.main()
