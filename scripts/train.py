#!/usr/bin/env python3
"""Execute the frozen small-scale public-data protocol; retain all outcomes."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from rejuvenation.pipeline import SEED,frozen_split,mlp_pipeline,read_dataset,ridge_pipeline,sha256


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(y,pred):
    return {'n':int(len(y)), 'mae_years':float(mean_absolute_error(y,pred)),
            'rmse_years':float(np.sqrt(mean_squared_error(y,pred))),
            'median_absolute_error_years':float(np.median(np.abs(y-pred))),
            'r2':float(r2_score(y,pred)) if len(y)>1 and np.var(y)>0 else None,
            'pearson_r':float(pearsonr(y,pred).statistic) if len(y)>2 and np.var(y)>0 and np.var(pred)>0 else None}


def lifecycle_smoke(out):
    """A tiny optimizer/update/save/load/resume check on a fixture, not a result."""
    rng=np.random.default_rng(SEED)
    x=rng.normal(size=(24,16));y=x[:,0]*2+x[:,1]
    m=MLPRegressor(hidden_layer_sizes=(8,),activation='tanh',solver='adam',random_state=SEED,max_iter=1)
    m.partial_fit(x,y)
    first=[c.copy() for c in m.coefs_]
    tmp=out/'smoke_checkpoint.joblib';joblib.dump(m,tmp)
    resumed=joblib.load(tmp)
    before=m.predict(x)
    np.testing.assert_allclose(before,resumed.predict(x),rtol=0,atol=1e-12)
    m.partial_fit(x,y);resumed.partial_fit(x,y)
    np.testing.assert_allclose(m.predict(x),resumed.predict(x),rtol=0,atol=1e-12)
    movement=sum(float(np.linalg.norm(a-b)) for a,b in zip(first,m.coefs_))
    if not movement>0 or not np.isfinite(m.loss_): raise ValueError('Optimizer lifecycle failed')
    tmp.unlink()
    return {'status':'PASS','fixture_samples':24,'network_trainable_parameters':sum(c.size for c in m.coefs_)+sum(c.size for c in m.intercepts_),
            'optimizer_parameter_movement_l2_sum':movement,'save_load_exact':True,'resume_matches_uninterrupted':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='pilot_20261021_v1');args=parser.parse_args()
    start=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    os.chdir(ROOT)
    out=ROOT/'results'/args.run_id;modeldir=ROOT/'models'/args.run_id
    out.mkdir(parents=True,exist_ok=False);modeldir.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    subprocess.run([sys.executable,str(ROOT/'scripts/download_data.py')],check=True)
    X,meta,annotation=read_dataset(ROOT/'data/raw')
    train,test=frozen_split(meta)
    manifest=meta.copy();manifest['split']='excluded';manifest.loc[train,'split']='train';manifest.loc[test,'split']='holdout'
    manifest.to_csv(out/'split_manifest.csv',index=False)
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    manifest.to_csv(ROOT/'data/processed/sample_manifest.csv',index=False)
    write_json(out/'protocol_snapshot.json',protocol)
    if len(set(meta.iloc[train].cell_id)&set(meta.iloc[test].cell_id)): raise ValueError('Split leakage')
    ytrain=meta.iloc[train].age_years.to_numpy();ytest=meta.iloc[test].age_years.to_numpy()
    gates={'preflight':{'status':'PASS','total_samples':len(meta),'healthy_adult_samples':len(train)+len(test),
           'train_samples':len(train),'holdout_samples':len(test),'public_cell_ID_overlap':0,'expression_features':X.shape[1]},'lifecycle_smoke':lifecycle_smoke(out)}
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always',ConvergenceWarning)
            canary=mlp_pipeline().fit(X[train[:24]],ytrain[:24])
            canpred=canary.predict(X[train[:24]])
            if not np.isfinite(canpred).all(): raise ValueError('Canary nonfinite predictions')
            gates['real_data_canary']={'status':'PASS','samples':24,'finite_predictions':True,
                 'loss':float(canary.regressor_.named_steps['regressor'].loss_),
                 'iterations':int(canary.regressor_.named_steps['regressor'].n_iter_),
                 'convergence_warnings':[str(w.message) for w in captured]}
        write_json(out/'preflight_gates.json',gates)
        print(f'Preflight PASS; frozen train={len(train)}, holdout={len(test)}. Full fit starts.',flush=True)
        cv=GroupKFold(n_splits=5,shuffle=True,random_state=SEED)
        cv_pairs=list(cv.split(X[train],ytrain,groups=meta.iloc[train].cell_id))
        for fold,(tr,va) in enumerate(cv_pairs):
            if set(meta.iloc[train[tr]].cell_id)&set(meta.iloc[train[va]].cell_id):raise ValueError('InnerCV cell-ID overlap')
        write_json(out/'inner_cv_manifest.json',[{'fold':i,'train_gsm':meta.iloc[train[tr]].gsm.tolist(),'validation_gsm':meta.iloc[train[va]].gsm.tolist()} for i,(tr,va) in enumerate(cv_pairs)])
        grid=GridSearchCV(ridge_pipeline(1),{'regressor__alpha':[1,10,100,1000]},scoring='neg_mean_absolute_error',
                          cv=cv_pairs,n_jobs=1,refit=True,error_score='raise')
        grid.fit(X[train],ytrain)
        pd.DataFrame(grid.cv_results_).to_csv(out/'ridge_inner_cv.csv',index=False)
        ridge=grid.best_estimator_
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always',ConvergenceWarning)
            mlp=mlp_pipeline().fit(X[train],ytrain)
            fitwarnings=[str(w.message) for w in captured]
        predictions={'median_baseline':np.full(len(test),np.median(ytrain)), 'ridge':ridge.predict(X[test]),'compact_mlp':mlp.predict(X[test])}
        if not all(np.isfinite(p).all() for p in predictions.values()):raise ValueError('Nonfinite holdout predictions')
        # Freeze predictions immediately; no later model fitting may use holdout feedback.
        predframe=meta.iloc[test][['gsm','cell_id','sample_title','age_years','sex','platform']].copy()
        for name,pred in predictions.items():predframe['prediction_'+name]=pred
        predframe.to_csv(out/'predictions.csv',index=False)
        for name,model in [('ridge',ridge),('compact_mlp',mlp)]:
            path=modeldir/(name+'.joblib');joblib.dump(model,path)
            loaded=joblib.load(path);np.testing.assert_allclose(loaded.predict(X[test]),predictions[name],rtol=0,atol=1e-12)
        rng=np.random.default_rng(20261022);bootstrap_idx=rng.integers(0,len(test),size=(2000,len(test)))
        errors={name:np.abs(ytest-pred) for name,pred in predictions.items()}
        metrics={'protocol_id':protocol['protocol_id'],'run_id':args.run_id,'frozen_protocol_sha256':sha256(ROOT/'configs/protocol.json'),
                 'sample_counts':{'total':len(meta),'normal':int((meta.disease=='Normal').sum()),'hgps_excluded':int((meta.disease=='HGPS').sum()),
                    'pediatric_normal_excluded':int(((meta.disease=='Normal')&(meta.age_years<18)).sum()),'healthy_adult':len(train)+len(test),'train':len(train),'holdout':len(test)},
                 'models':{},'mlp_vs_ridge':{},'subgroups':{},'interpretation':'Chronological-age association on one public cohort only; no biological rejuvenation or causal mechanism demonstrated.'}
        for name,pred in predictions.items():
            record=summarize(ytest,pred)
            record['bootstrap_mae_ci95']=np.quantile(errors[name][bootstrap_idx].mean(axis=1),[.025,.975]).tolist()
            if name!='median_baseline':
                delta=(errors['median_baseline']-errors[name])[bootstrap_idx].mean(axis=1)
                record['paired_improvement_vs_baseline_ci95']=np.quantile(delta,[.025,.975]).tolist()
                record['relative_mae_reduction_vs_baseline']=1-record['mae_years']/float(errors['median_baseline'].mean())
                record['criterion']='PASS' if record['relative_mae_reduction_vs_baseline']>=.2 and record['paired_improvement_vs_baseline_ci95'][0]>0 else 'FAIL'
            metrics['models'][name]=record
        delta=(errors['ridge']-errors['compact_mlp'])[bootstrap_idx].mean(axis=1)
        metrics['mlp_vs_ridge']={'paired_mae_improvement_years':float((errors['ridge']-errors['compact_mlp']).mean()),
                 'paired_improvement_ci95':np.quantile(delta,[.025,.975]).tolist(),
                 'superiority_criterion':'PASS' if np.quantile(delta,.025)>0 else 'FAIL'}
        for attr in ['sex','platform','age_band']:
            groups=meta.iloc[test][attr].to_numpy() if attr!='age_band' else np.where(ytest<60,'18-59','60+')
            for group in sorted(set(groups)):
                ix=groups==group
                metrics['subgroups'][attr+'='+group]={name:summarize(ytest[ix],pred[ix]) for name,pred in predictions.items()}
        print('Frozen holdout MAE:',{name:round(r['mae_years'],4) for name,r in metrics['models'].items()},flush=True)
        selected_alpha=float(grid.best_params_['regressor__alpha'])
        controls=[];rng=np.random.default_rng(20261023)
        for i in range(30):
            if time.monotonic()-start>15*60:raise TimeoutError('Predeclared CPU wall budget exceeded')
            shuffled=rng.permutation(ytrain)
            control=ridge_pipeline(selected_alpha).fit(X[train],shuffled)
            controls.append({'permutation':i+1,'mae_years':float(mean_absolute_error(ytest,control.predict(X[test])))})
        pd.DataFrame(controls).to_csv(out/'negative_control.csv',index=False)
        metrics['training_label_shuffle_diagnostic']={'permutations':len(controls),'alpha':selected_alpha,
                  'mae_mean':float(np.mean([r['mae_years'] for r in controls])),
                  'mae_min':float(min(r['mae_years'] for r in controls)),
                  'mae_max':float(max(r['mae_years'] for r in controls)),
                  'note':'Diagnostic only: fixed train-selected alpha; not a confirmatory permutation p-value.'}
        metrics['ridge_selected_alpha']=selected_alpha
        metrics['mlp_details']={'hidden_layer_sizes':[8],'pca_components':16,
              'network_trainable_parameters':sum(c.size for c in mlp.regressor_.named_steps['regressor'].coefs_)+sum(c.size for c in mlp.regressor_.named_steps['regressor'].intercepts_),
              'iterations':int(mlp.regressor_.named_steps['regressor'].n_iter_),'training_scaled_target_loss':float(mlp.regressor_.named_steps['regressor'].loss_),
              'convergence_warnings':fitwarnings,'checkpoint_reloaded_prediction_exact':True}
        metrics['ridge_checkpoint_reloaded_prediction_exact']=True
        write_json(out/'metrics.json',metrics)
        # Association coefficients are not intervention targets or causal rankings.
        selected=ridge.named_steps['features'].selected_indices_
        coef=annotation.iloc[selected].copy();coef['ridge_coefficient_per_training_standard_deviation']=ridge.named_steps['regressor'].coef_
        coef['interpretation']='correlational fitted model weight; not a rejuvenation target'
        coef.to_csv(out/'ridge_feature_weights.csv',index=False)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for name,color in [('ridge','#285F96'),('compact_mlp','#AF5B33')]:
            axes[0].scatter(ytest,predictions[name],label=name,s=28,alpha=.8,c=color)
        axes[0].plot([18,100],[18,100],'k--',linewidth=1);axes[0].set(xlabel='Chronological age (years)',ylabel='Held-out prediction (years)',title=f'Public healthy adults: {len(test)} held-out cell lines');axes[0].legend(fontsize=8)
        names=list(predictions);values=[metrics['models'][n]['mae_years'] for n in names]
        cis=np.array([metrics['models'][n]['bootstrap_mae_ci95'] for n in names]);errors_ci=np.array([np.array(values)-cis[:,0],cis[:,1]-np.array(values)])
        axes[1].bar(names,values,color=['#777777','#285F96','#AF5B33']);axes[1].errorbar(names,values,yerr=errors_ci,fmt='none',ecolor='black',capsize=4)
        axes[1].set(ylabel='Holdout MAE (years)',title='95% sample-bootstrap intervals');axes[1].tick_params(axis='x',labelsize=8)
        fig.suptitle('Exploratory chronological-age prediction — no rejuvenation experiment',fontsize=11)
        fig.savefig(out/'holdout_results.png',dpi=180);fig.savefig(out/'holdout_results.pdf');plt.close(fig)
    wall=time.monotonic()-start
    code_files=['src/rejuvenation/pipeline.py','src/rejuvenation/__init__.py','scripts/train.py','scripts/download_data.py','scripts/predict.py','tests/test_pipeline.py','configs/protocol.json','configs/requirements.lock.txt']
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError:commit=None
    runmanifest={'run_id':args.run_id,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':wall,
          'git_commit_at_run':commit,'source_status':'Uncommitted initial repository; exact executed source hashes below are authoritative.',
          'command':f'python scripts/train.py --run-id {args.run_id}',
          'runtime':{'python':platform.python_version(),'platform':platform.platform(),'cpu_threads_limited':1,'gpu_used':False,
               'packages':{n:md.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib','threadpoolctl']}},
          'source_sha256':{f:sha256(ROOT/f) for f in code_files},
          'data_sources':json.loads((ROOT/'data/raw/source_manifest.json').read_text()),
          'result_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
          'model_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(modeldir.iterdir()) if p.is_file()},
          'cost':{'cloud_usd':0,'gpu_minutes':0,'cpu_wall_minutes':wall/60}}
    write_json(out/'run_manifest.json',runmanifest)
    print(json.dumps({'run_id':args.run_id,'wall_seconds':wall,'models':metrics['models'],'mlp_vs_ridge':metrics['mlp_vs_ridge']},indent=2),flush=True)

if __name__=='__main__':main()
