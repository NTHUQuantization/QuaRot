"""Synthetic CPU fixtures only; no benchmark measurements or GPU use."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare
import measure
import run
import summarize


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,label='custom-implementation',token=124):
        root=self.root/label;root.mkdir()
        for name in ('prompts.jsonl','profile.json'):
            (root/name).write_bytes((run.HERE/'reference'/name).read_bytes())
        config=dict(label=label,run_id=label+'-001')
        contract=run.contract(config,dict(target={},tokenizer={}))
        (root/'contract.json').write_text(json.dumps(contract))
        prompts=[p for p in summarize.read(root/'prompts.jsonl') if p['stage']=='pilot']
        def write(name,rows):(root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        rows=[];qualification=[];gpu=[]
        for p in prompts:
            for sweep in range(3):
                gpu.append(dict(returncode=0,gpu_processes=[['synthetic',1,1]]))
                for phase in summarize.PHASES:
                    row=dict(label=label,run_id=config['run_id'],stage='pilot',dataset=p['dataset'],sample_id=p['sample_id'],
                        sweep=sweep,phase=phase,valid=True,gpu_competition=False,elapsed_ms=10+sweep,
                        input_hash=p['input_hash'],input_tokens=len(p['input_ids']),
                        peak_reserved_bytes=3,peak_allocated_bytes=2,baseline_allocated_bytes=1,
                        incremental_peak_allocated_bytes=1)
                    if phase!='prefill':
                        ids=[token,151645]
                        row.update(output_ids=ids,output_tokens=2,stop_reason='eos',cap_reached=False,
                            output_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),raw_stats=dict(emitted_tokens_per_step=[1,1]))
                        qualification.append(dict(dataset=p['dataset'],sample_id=p['sample_id'],sweep=sweep,
                            phase=phase,within_run_exact=True))
                    rows.append(row)
        write('runs.jsonl',rows);write('qualification.jsonl',qualification);write('gpu.jsonl',gpu)
        (root/'complete.json').write_text(json.dumps(dict(stage='pilot',label=label,records=108,all_within_run_exact=True)))
        (root/'manifest.json').write_text(json.dumps(dict(label=label,stage='pilot',cap=128,repeats=3,
            eos_ids=[151645],prompt_file_sha256=run.digest(root/'prompts.jsonl'),contract=contract,
            worker_sha256=contract['common']['worker_sha256'],adapter_sha256=contract['adapter_sha256'],
            torch='synthetic',hip='synthetic',python='synthetic',gpu='synthetic',dependencies={},host_cpu='synthetic')))
        return root

    def test_frozen_orders(self):
        prompts=summarize.read(run.HERE/'reference/prompts.jsonl')
        manifest=json.loads((run.HERE/'reference/profile.json').read_text())
        for stage,n in [('pilot',12),('formal',96)]:
            ids={i for i,p in enumerate(prompts) if p['stage']==stage}
            self.assertEqual(len(ids),n)
            self.assertEqual(len(manifest['orders'][stage]),3)
            for order in manifest['orders'][stage]:
                self.assertEqual(len(order),n);self.assertEqual(set(order),ids)
            for dataset in summarize.DATASETS:
                self.assertEqual(sum(prompts[i]['dataset']==dataset for i in ids),n//3)

    def test_percentiles_and_independent_summary(self):
        for q,wanted in [(.1,3),(.9,27),(.99,29.7)]:
            self.assertAlmostEqual(summarize.quantile([0,10,20,30],q),wanted)
        root=self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):summarize.main(root)
        self.assertEqual(json.loads((root/'validation.json').read_text())['records'],108)

    def test_arbitrary_labels_and_output_difference(self):
        roots=[self.fixture('version-one'),self.fixture('another-kernel',token=123)]
        ratios,outputs=compare.compare(roots)
        self.assertTrue(all(r['identical_output_prompts']==0 for r in ratios))
        self.assertTrue(all(r['interpretation'].startswith('natural-generation') for r in ratios))
        self.assertTrue(all(r['first_difference']==0 for r in outputs))

    def test_matching_outputs(self):
        roots=[self.fixture('version-one'),self.fixture('version-two')]
        ratios,_=compare.compare(roots)
        self.assertTrue(all(r['interpretation']=='equal-output speedup' for r in ratios))
        self.assertTrue(all(r['paired_geomean_e2e_time_ratio']==1 for r in ratios))

    def test_environment_mismatch(self):
        roots=[self.fixture('first'),self.fixture('second')]
        path=roots[1]/'manifest.json';m=json.loads(path.read_text());m['gpu']='other-gpu';path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Environment differs'):compare.compare(roots)
        ratios,_=compare.compare(roots,True)
        self.assertTrue(all(not r['environment_matches'] for r in ratios))

    def test_contract_mismatch(self):
        roots=[self.fixture('first'),self.fixture('second')]
        path=roots[1]/'contract.json';c=json.loads(path.read_text());c['common']['target_sha256']={'different':'hash'}
        c['inputs']['target']=c['common']['target_sha256'];c['comparison_id']=run.identity(c['common'])
        path.write_text(json.dumps(c))
        path=roots[1]/'manifest.json';m=json.loads(path.read_text());m['contract']=c;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Incompatible'):compare.compare(roots)

    def test_duplicates_rejected(self):
        root=self.fixture();path=root/'runs.jsonl'
        rows=summarize.read(path);rows[1]=rows[0];path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(AssertionError):summarize.audit(root)

    def test_within_run_mismatch_rejected(self):
        root=self.fixture();path=root/'runs.jsonl';rows=summarize.read(path)
        rows[1]['output_ids'][0]=123
        rows[1]['output_hash']=hashlib.sha256(json.dumps(rows[1]['output_ids']).encode()).hexdigest()
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(AssertionError,'Within-run'):summarize.audit(root)

    def test_invalid_memory_rejected(self):
        root=self.fixture();path=root/'runs.jsonl';rows=summarize.read(path);rows[0]['peak_allocated_bytes']=0
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(AssertionError):summarize.audit(root)

    def test_rtn_rejected(self):
        from require_gptq import require_gptq_checkpoint
        (self.root/'config.json').write_text(json.dumps(dict(quarot_conversion=dict(method='rtn',w_bits=4))))
        with self.assertRaisesRegex(ValueError,'GPTQ'):require_gptq_checkpoint(self.root)

    def test_custom_repo_and_unsupported_profile_key(self):
        c=json.loads((run.HERE/'config.example.json').read_text());c['repo']='any-branch-checkout'
        path=self.root/'config.json';path.write_text(json.dumps(c))
        self.assertEqual(run.read_config(path)['repo'],str(self.root/'any-branch-checkout'))
        c['batch_size']=2;path.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'Unknown config'):run.read_config(path)

    def test_worker_end_to_end_with_mock_cuda(self):
        """Execute the real 108-row worker through a non-GPU adapter and audit it."""
        output=self.root/'worker';output.mkdir()
        repo=self.root/'source';repo.mkdir()
        config=dict(label='custom-mock',run_id='mock-001',repo=str(repo),source_commit='synthetic',source_branch='synthetic')
        bound=run.contract(config,dict(target={},tokenizer={}))
        bound['adapter_sha256']=run.digest(__file__)
        (output/'contract.json').write_text(json.dumps(bound))
        for name in ('prompts.jsonl','profile.json'):
            (output/name).write_bytes((run.HERE/'reference'/name).read_bytes())
        events=[]
        class Phase:
            generated_source='synthetic phase for CPU integration test'
            def begin(self):
                events.append('prefix')
                return SimpleNamespace(close=lambda:None)
            def finish(self,state):
                events.append('decode')
                return SimpleNamespace(output_ids=[124,151645],emitted_tokens_per_step=[1,1])
        runtime=SimpleNamespace(repo=repo,eos_ids={151645},_verification_graphs={})
        adapter=SimpleNamespace(__file__=__file__,load=lambda *_:runtime,
            phases=lambda *_:Phase(),provenance=lambda _:dict(synthetic=True))
        cuda=SimpleNamespace(mem_get_info=lambda:(32<<30,32<<30),get_device_properties=lambda _: 'synthetic',
            synchronize=lambda:events.append('sync'),reset_peak_memory_stats=lambda:events.append('reset'),
            memory_allocated=lambda:1,memory_reserved=lambda:3,max_memory_allocated=lambda:2,max_memory_reserved=lambda:3)
        torch=SimpleNamespace(__version__='synthetic',version=SimpleNamespace(hip='synthetic'),cuda=cuda,
            manual_seed=lambda _:None,set_num_threads=lambda _:None,tensor=lambda x,**kw:x,long='long')
        gpu=dict(unix=0,returncode=0,text='synthetic',stderr='',gpu_processes=[['mock',1,1]])
        args=SimpleNamespace(output=output,stage='pilot',run_id='mock-001')
        with patch.dict('sys.modules',{'torch':torch}),patch('importlib.metadata.version',return_value='synthetic'),\
             patch.object(measure,'gpu_snapshot',return_value=gpu),contextlib.redirect_stdout(io.StringIO()):
            measure.main(args,config,adapter,bound)
            self.assertFalse((output/'complete.json').exists())
            (output/'measurement_complete.json').replace(output/'complete.json')
            summarize.main(output)
        self.assertEqual(len(summarize.read(output/'runs.jsonl')),108)
        self.assertEqual(events.count('reset'),108)
        self.assertEqual(json.loads((output/'validation.json').read_text())['prompts'],12)


if __name__=='__main__':unittest.main()
