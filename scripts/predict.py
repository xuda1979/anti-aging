#!/usr/bin/env python3
"""Example only: predict held-out public samples, no clinical interpretation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
from rejuvenation.pipeline import read_dataset

p=argparse.ArgumentParser();p.add_argument('--model',default='models/pilot_20261021_v2/ridge.joblib');p.add_argument('--sample',required=True);args=p.parse_args()
X,meta,_=read_dataset('data/raw')
idx=meta.index[meta.gsm==args.sample].tolist()
if len(idx)!=1: raise SystemExit('Supply exactly one known public GSM accession')
print({'sample':args.sample,'predicted_chronological_age_years':float(joblib.load(args.model).predict(X[idx])[0]),'warning':'Research only; not biological age, function or rejuvenation.'})
