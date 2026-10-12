#!/usr/bin/env python
"""Watch the authorized E20 checkpoints and evaluate their fixed banks in order."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,default=Path('/root/.local/share/coopdiff/coupled_research20_seed73'))
    parser.add_argument('--bank-root',type=Path,default=Path('/root/.local/share/coopdiff/small_sample_seed73/coupled_pilot'))
    args = parser.parse_args()
    args.research_root.mkdir(parents=True,exist_ok=True)
    (args.research_root/'evaluation_watcher.pid').write_text(str(os.getpid()))
    env = os.environ.copy()
    env.pop('GCC_ROOT',None)
    env.update(CUDA_VISIBLE_DEVICES='3', HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TRICOOPDIFF_CACHE_DIR='/data_yutian/mwj/CoopDiff/results/runtime_cache/diffdock')
    entry = Path(__file__).resolve().with_name('evaluate_coupled_checkpoint.py')
    print(json.dumps({'event':'watcher_started','pid':os.getpid(),
        'updates':[5,10,20], 'physical_gpu':3}),flush=True)
    for update in (5,10,20):
        checkpoint=args.research_root/f'checkpoint_update{update}.pt'
        print(json.dumps({'event':'waiting','update':update}),flush=True)
        while not checkpoint.is_file():
            time.sleep(10)
        weights='raw,ema' if update==20 else 'raw'
        command=[sys.executable,'-u',str(entry),'--checkpoint',str(checkpoint),
            '--bank-root',str(args.bank_root),'--output-dir',str(args.research_root/'evaluation'),
            '--weights',weights]
        print(json.dumps({'event':'evaluation_started','update':update,'weights':weights}),flush=True)
        subprocess.run(command,cwd='/tmp',env=env,check=True)
        print(json.dumps({'event':'evaluation_completed','update':update}),flush=True)
    print(json.dumps({'event':'watcher_complete','last_update':20}),flush=True)


if __name__=='__main__':
    main()
