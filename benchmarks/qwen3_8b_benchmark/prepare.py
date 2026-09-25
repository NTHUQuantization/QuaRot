"""Build the selected runtime and its FHT dependency, without changing Git branches."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from run import read_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=read_config(args.config)
    if config.get('adapter'):raise ValueError('Build custom runtimes with their own build instructions')
    env=dict(os.environ, QUAROT_HIP_ARCHS='gfx1201', HADACORE_HIP_ARCHS='gfx1201',
             HADACORE_ENABLE_EXPERIMENTAL_WMMA='1',HADACORE_FORCE_GFX12_WMMA='1')
    for repo in (Path(config['repo']),Path(config['fht_path'])):
        if not (repo/'setup.py').exists():raise ValueError(f'No setup.py: {repo}')
        subprocess.run([sys.executable,'setup.py','build_ext','--inplace','--force'],cwd=repo,env=env,check=True)
    print('Selected runtime and FHT extensions rebuilt in-place.')


if __name__=='__main__':main()
