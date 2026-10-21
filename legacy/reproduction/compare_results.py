#!/usr/bin/env python3
"""Compare byte-preserved historical outputs with an explicitly supplied replay copy.

This comparison is supporting evidence; independent verification is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FILES = [
 'public_association_genes.csv','public_marker_changes.csv','public_predictions.csv',
 'public_permutation_null.csv','synthetic_runs.csv','synthetic_rankings.csv',
 'synthetic_summary.csv','synthetic_ranking_summary.csv','prospective_power.csv',
 'public_summary.json','synthetic_metadata.json',
]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare_json(a,b,path=''):
    if isinstance(a,dict):
        if set(a)!=set(b):return ['keys:'+path]
        return sum([compare_json(a[k],b[k],path+'/'+str(k)) for k in a],[])
    if isinstance(a,list):
        if len(a)!=len(b):return ['length:'+path]
        return sum([compare_json(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
    if isinstance(a,(int,float)) and not isinstance(a,bool):return [] if np.isclose(a,b,rtol=1e-10,atol=1e-12) else ['numeric:'+path]
    return [] if a==b else ['value:'+path]

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--replay',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    archive,replay=Path(args.archive),Path(args.replay);checks=[]
    for name in FILES:
        a,b=archive/'results'/name,replay/'results'/name
        record={'file':name,'archive_sha256':digest(a),'replay_sha256':digest(b),'byte_identical':a.read_bytes()==b.read_bytes()}
        try:
            if name.endswith('.json'):
                discrepancies=compare_json(json.loads(a.read_text()),json.loads(b.read_text()))
                if discrepancies:raise ValueError(str(discrepancies))
            elif name=='synthetic_ranking_summary.csv':
                pd.testing.assert_frame_equal(pd.read_csv(a,header=[0,1],index_col=[0,1]),pd.read_csv(b,header=[0,1],index_col=[0,1]),check_exact=False,check_dtype=False,rtol=1e-10,atol=1e-12)
            else:
                pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b),check_exact=False,check_dtype=False,rtol=1e-10,atol=1e-12)
            record['status']='PASS'
        except Exception as e:record.update(status='FAIL',error=str(e))
        checks.append(record)
    record={'numeric_tolerance':{'relative':1e-10,'absolute':1e-12},'files':checks,'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'}
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    return 0 if record['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
