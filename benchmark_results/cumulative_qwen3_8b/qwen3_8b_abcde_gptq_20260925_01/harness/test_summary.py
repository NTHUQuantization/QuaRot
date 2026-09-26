"""Synthetic temporary fixtures test audits; never used as benchmark results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import summarize


class SummaryAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='abcde_summary_test_')
        self.addCleanup(self.temp.cleanup)
        self.original_root=summarize.ROOT
        self.addCleanup(setattr,summarize,'ROOT',self.original_root)
        root=Path(self.temp.name)
        source=self.original_root.parent.parent/'fused_v1/ablation_qwen3_8b_cd_20260925_01'
        (root/'prompts.jsonl').write_bytes((source/'prompts.jsonl').read_bytes())
        old=summarize.read(source/'pilot/runs.jsonl')
        for v in 'ABCDE':
            folder=root/'results/pilot'/v
            folder.mkdir(parents=True)
            rows=[]
            for r in old:
                if r['variant']!=('D' if v=='E' else 'C'):continue
                r=copy.deepcopy(r)
                r.update(variant=v,stage='pilot',baseline_allocated_bytes=1,
                         peak_allocated_bytes=2,peak_reserved_bytes=3,incremental_peak_allocated_bytes=1)
                if v=='A' and r['phase']!='prefill':
                    # A is internally reproducible but differs from the D oracle.
                    r['output_ids'][0]=123
                if r['phase']!='prefill':
                    r['output_hash']=hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest()
                rows.append(r)
            (folder/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (folder/'complete.json').write_text(json.dumps(dict(records=108,all_within_variant_exact=True)))
            (folder/'variant_manifest.json').write_text(json.dumps(dict(variant=v,stage='pilot',cap=128,
                eos_ids=[151643,151645],prompt_file_sha256='fixture',source_sha256={},
                runtime_provenance=dict(runtime_flags={}))))
            (folder/'qualification.jsonl').write_text((json.dumps(dict(within_variant_exact=True))+'\n')*72)
            (folder/'gpu.jsonl').write_text((json.dumps(dict(returncode=0,gpu_processes=[['fixture',1,1]]))+'\n')*36)
        summarize.ROOT=root

    def test_complete_and_cross_differences_retained(self):
        self.assertAlmostEqual(summarize.quantile([0,10,20,30],.1),3)
        self.assertAlmostEqual(summarize.quantile([0,10,20,30],.9),27)
        summarize.main('pilot')
        audit=json.loads((summarize.ROOT/'results/pilot/validation.json').read_text())
        self.assertEqual(audit['records'],540)
        self.assertEqual(audit['cross_pairs']['AD'],0)
        self.assertEqual(audit['cross_pairs']['DE'],12)

    def test_duplicate_measurement_rejected(self):
        path=summarize.ROOT/'results/pilot/A/runs.jsonl'
        rows=summarize.read(path)
        rows[1]=rows[0]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(AssertionError):summarize.main('pilot')

    def test_invalid_memory_rejected(self):
        path=summarize.ROOT/'results/pilot/A/runs.jsonl'
        rows=summarize.read(path)
        rows[0]['peak_allocated_bytes']=0
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(AssertionError):summarize.main('pilot')


if __name__=='__main__':unittest.main()
