"""Run one GPU worker at a time; preserve failed runs and stop on any error."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONTAINER_ROOT = '/workspace_root/benchmark_work/' + ROOT.name
RUN_ID = 'qwen3_8b_abcde_gptq_20260925_01'


def main():
    assert not (ROOT/'GPTQ_REQUIRED.json').exists(), 'Run is administratively blocked'
    subprocess.run([sys.executable,str(ROOT/'initialize.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'verify_label_migration.py')],check=True)
    if '--adopt-formal-a' in sys.argv:
        initial=ROOT/'results/formal/A'
        while not (initial/'complete.json').exists():
            status=subprocess.run(['docker','exec','qwen3_32b_quarot_clean','pgrep','-f',
                '^python measure.py --variant A --stage formal '],capture_output=True,text=True)
            assert status.returncode==0 or (initial/'complete.json').exists(), 'Active formal A exited without completion'
            time.sleep(15)
    # Optional adoption is unused for the new GPTQ run.
    initial = ROOT/'results/pilot/A'
    if '--adopt-initial' in sys.argv and not (initial/'complete.json').exists():
        while not (initial/'complete.json').exists():
            status = subprocess.run(['docker','exec','qwen3_32b_quarot_clean','pgrep','-f',
                '^python measure.py --variant A --stage pilot '],capture_output=True,text=True)
            if status.returncode:
                assert (initial/'complete.json').exists(), 'Initial A pilot exited without completion'
            time.sleep(15)
    for stage in ('pilot', 'formal'):
        for variant in 'DABCE':
            out = ROOT/'results'/stage/variant
            if (out/'complete.json').exists():
                result = json.loads((out/'complete.json').read_text())
                assert result['records'] == (108 if stage == 'pilot' else 864)
                print('Already complete:', stage, variant, flush=True)
                continue
            assert not (out/'runs.jsonl').exists(), f'Partial run needs review: {out}'
            out.mkdir(parents=True, exist_ok=True)
            command = ['docker', 'exec', '-w', CONTAINER_ROOT]
            env = dict(OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                       QUAROT_BATCHED_H128='1', QUAROT_STATIC_KV_METADATA='1',
                       QUAROT_VERIFICATION_GRAPH='1', QUAROT_CHUNK_PREPROCESS='1',
                       GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory',
                       GIT_CONFIG_VALUE_0='/workspace_root/fused_v1')
            for key, value in env.items():
                command += ['-e', key+'='+value]
            command += ['qwen3_32b_quarot_clean', 'python', 'measure.py',
                        '--variant', variant, '--stage', stage, '--output',
                        f'{CONTAINER_ROOT}/results/{stage}/{variant}', '--run-id', RUN_ID]
            (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            print('Starting:', stage, variant, time.time(), flush=True)
            with (out/'worker.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert (out/'complete.json').exists()
            print('Completed:', stage, variant, flush=True)
        subprocess.run([sys.executable,str(ROOT/'summarize.py'),stage],check=True)
    subprocess.run([sys.executable,str(ROOT/'publish_results.py')],check=True)
    print('All five variants validated and published.', flush=True)


if __name__ == '__main__':
    state = dict(started_unix=time.time(),status='running')
    (ROOT/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
    try:
        main()
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_unix=time.time())
        raise
    else:
        state.update(status='complete',finished_unix=time.time())
    finally:
        (ROOT/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
