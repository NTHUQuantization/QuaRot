"""CPU guards for source replacement while an unchanged formal worker is active."""
from pathlib import Path
import unittest
from unittest.mock import patch

import measure
import run_queue
import verify_source_migration as guard

ROOT=Path(__file__).resolve().parent


class SourceMigrationTests(unittest.TestCase):
    def test_pinned_mapping_and_next_steps(self):
        guard.verify(ROOT)
        plan=run_queue.pending_plan()
        desired=[('pilot','B'),('pilot','C'),('formal','B'),('formal','C'),('formal','E')]
        pending=[(stage,v) for stage,v in desired if not (ROOT/'results'/stage/v/'complete.json').exists()]
        self.assertEqual([(s['stage'],s['variant']) for s in plan['steps']],pending)
        for step in plan['steps']:
            if step['variant']=='B':self.assertEqual(step['commit'],'973a13dd849cbf2cee1e134919356ce71e23cdb5')
            if step['variant']=='C':self.assertEqual(step['commit'],'6c30374901038138ac5094e6eaed35a5ca35db2f')

    def test_wrong_commit_is_rejected(self):
        original=guard.subprocess.check_output
        def wrong(command,**kwargs):
            if 'rev-parse' in command and any('B_GEMM' in str(x) for x in command):
                return 'ce73a82a2c7c4ed258dcd4e6e69ad9b2515aef52\n'
            return original(command,**kwargs)
        with patch.object(guard.subprocess,'check_output',side_effect=wrong):
            with self.assertRaises(AssertionError):guard.verify(ROOT,'B')

    def test_wrong_binary_is_rejected(self):
        original=measure.snapshot
        def wrong(repo):
            data=original(repo)
            if repo.name=='C_Hadacore':
                key=next(n for n in data if n.endswith('.so'))
                data[key]='wrong-binary'
            return data
        with patch.object(measure,'snapshot',side_effect=wrong):
            with self.assertRaisesRegex(AssertionError,'Pinned source/binary changed'):guard.verify(ROOT,'C')


if __name__=='__main__':unittest.main()
