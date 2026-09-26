"""Portable replay of one variant using an explicit checkout and checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess

import runtime_adapter as adapter
import measure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=list('ABCDE'),required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--draft',type=Path)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--stage',choices=('pilot','formal'),default='formal')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',default='replay')
    args=parser.parse_args()
    assert args.variant!='E' or args.draft is not None
    runroot=args.output.resolve()
    args.output=runroot/'results'/args.stage/args.variant
    assert not args.output.exists(), 'This variant/stage already exists; choose a new replay root'
    runroot.mkdir(parents=True,exist_ok=True)
    adapter.BRANCHES[args.variant]=args.repo.resolve()
    adapter.TARGET=args.target.resolve()
    adapter.TOKENIZER=args.tokenizer.resolve()
    if args.draft:adapter.DRAFT=args.draft.resolve()
    measure.OLD=runroot
    import shutil
    for name in ('prompts.jsonl','protocol_manifest.json'):
        shutil.copy2(args.reference.resolve()/name,runroot/name)
    measure.ROOT=runroot
    git=lambda *items:subprocess.check_output(['git','-c','safe.directory='+str(args.repo.resolve()),'-C',str(args.repo.resolve()),*items],text=True).strip()
    metadata={args.variant:dict(commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'))}
    (runroot/'host_git_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    import torch
    with torch.inference_mode():measure.main(args)


if __name__=='__main__':
    main()
