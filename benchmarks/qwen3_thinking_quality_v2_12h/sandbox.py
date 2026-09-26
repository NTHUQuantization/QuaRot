"""HumanEval isolation: candidate timeout starts inside the ready container."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

SUPERVISOR = '''import json,subprocess,time
started=time.monotonic()
try:
    r=subprocess.run(["python","-I","-B","/candidate.py"],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=10)
    result={"exit_code":r.returncode,"timed_out":False,"stderr_tail":r.stderr[-2000:].decode("utf-8",errors="replace")}
except subprocess.TimeoutExpired:
    result={"exit_code":124,"timed_out":True,"stderr_tail":"candidate exceeded 10 seconds"}
result["execution_seconds"]=time.monotonic()-started
print(json.dumps(result))
'''

def atomic(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
def execute(path,image):
    name='qwen-full-he-'+uuid.uuid4().hex[:16]; stat=path.stat()
    cmd=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','32','--memory','128m','--memory-swap','128m',
        '--cpus','1','--user',f'{stat.st_uid}:{stat.st_gid}','--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
        '--mount',f'type=bind,src={path.resolve()},dst=/candidate.py,readonly',image,
        'python','-I','-B','-c',SUPERVISOR]
    start=time.monotonic()
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    except subprocess.TimeoutExpired:
        # Infrastructure timeout is an error, never a failed candidate score.
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
        raise RuntimeError(f'Container startup/transport timeout for {path.name}')
    if r.returncode:
        raise RuntimeError(f'Container infrastructure exit {r.returncode}: {r.stderr[-2000:]}')
    result=json.loads(r.stdout)
    result.update(elapsed_seconds=time.monotonic()-start,candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return path.name,result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--candidates-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); candidates=sorted(a.candidates_dir.glob('*.py'))
    if not candidates or any(not re.fullmatch(r's\d+_p\d{3}\.py',p.name) for p in candidates): raise ValueError('Invalid candidates')
    image=subprocess.check_output(['docker','image','inspect','python@sha256:6857d2dae63e052057f2db389a7061188ac9a92a3fa8d402bde68f36df6fada1','--format','{{.Id}}'],text=True).strip()
    statuses=json.loads(a.output.read_text()) if a.output.exists() else {}
    for path in candidates:
        if path.name in statuses and statuses[path.name]['candidate_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError('Candidate changed on sandbox resume')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(execute,path,image) for path in candidates if path.name not in statuses]
        for future in as_completed(futures):
            name,status=future.result(); statuses[name]=status; atomic(a.output,dict(sorted(statuses.items())))
    atomic(a.output.with_name('sandbox_provenance.json'),{'image':image,'candidate_timeout_seconds':10,
        'container_timeout_seconds':180,'supervisor':SUPERVISOR,'count':len(statuses)})
    print(json.dumps({'count':len(statuses),'passed':sum(r['exit_code']==0 for r in statuses.values()),
        'timed_out':sum(r['timed_out'] for r in statuses.values())}))
if __name__=='__main__': main()
