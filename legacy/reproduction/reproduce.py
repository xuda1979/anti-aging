#!/usr/bin/env python3
"""Replay a byte-preserved historical project in a temporary copy; no archive edits."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime,timezone

OUTPUT_FILES=['public_association_genes.csv','public_marker_changes.csv','public_predictions.csv','public_permutation_null.csv','synthetic_runs.csv','synthetic_rankings.csv','synthetic_summary.csv','synthetic_ranking_summary.csv','prospective_power.csv','public_summary.json','synthetic_metadata.json','runtime_all.json']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'CellClockAI'
    env=os.environ.copy();env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    records=[]
    with tempfile.TemporaryDirectory(prefix='cellclock-replay-') as temp:
        work=Path(temp)/'CellClockAI';shutil.copytree(source,work)
        stages=[('unit_tests',[sys.executable,'-m','unittest','discover','-s','tests','-v'],60),('scientific_replay',[sys.executable,'scripts/run_all.py','--only','all','--seeds','20','--permutations','100'],600),('prospective_power',[sys.executable,'scripts/power_analysis.py'],30)]
        if importlib.util.find_spec('torch') is not None:stages.append(('optional_graph_smoke',[sys.executable,'scripts/smoke_deep.py'],60))
        for name,cmd,timeout in stages:
            print('Starting',name,flush=True);start=time.monotonic()
            with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            records.append({'name':name,'command':cmd,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'completed_utc':datetime.now(timezone.utc).isoformat()})
            (out/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
            if r.returncode:raise SystemExit(f'{name} failed; retained log in {out}')
        outputs=out/'outputs';outputs.mkdir()
        for name in OUTPUT_FILES:shutil.copy2(work/'results'/name,outputs/name)
        shutil.copy2(work/'data/processed/public_sample_manifest.csv',outputs/'public_sample_manifest.csv')
        compare=Path(__file__).with_name('compare_results.py')
        subprocess.run([sys.executable,str(compare),'--archive',str(source),'--replay',str(work),'--output',str(out/'comparison.json')],check=True)
        (out/'optional_graph_status.json').write_text(json.dumps({'torch_available':importlib.util.find_spec('torch') is not None,'biologically_trained_graph_model':False,'data':'Optional architecture uses invented tensors only.'},indent=2)+'\n')
    print(f'Replay complete; fresh evidence: {out}')

if __name__=='__main__':main()
