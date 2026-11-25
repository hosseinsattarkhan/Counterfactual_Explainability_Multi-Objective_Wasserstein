"""Reproducible proof-of-concept operational screening-policy choice on real WDBC observations.
NOT a clinically validated screening protocol; thresholds and costs are illustrative.
"""
import sys,json,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ambiguity.wasserstein import DiscreteWasserstein, wasserstein_uniform_gap
from algorithms.exact_radius_v9 import exact_stability_radius
from algorithms.stability_v8 import certified_radius_bracket,audit_radius

def run():
    data=load_breast_cancer();X=data.data;y=(data.target==0).astype(int)
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
    thresholds=np.array([.10,.20,.30,.40,.50,.60,.70,.80])
    costs=np.zeros((len(thresholds),5,3));meta=[]
    for s,(train,test) in enumerate(folds.split(X,y)):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=2026))
        clf.fit(X[train],y[train]);prob=clf.predict_proba(X[test])[:,1]; yt=y[test]
        for k,t in enumerate(thresholds):
            refer=prob>=t
            fn=np.sum((~refer)&(yt==1))/max(1,np.sum(yt==1))
            fp=np.sum(refer&(yt==0))/max(1,np.sum(yt==0))
            rate=np.mean(refer)
            costs[k,s]=[fn,fp,rate]
        meta.append({'fold':s,'n_test':len(test),'prevalence':float(yt.mean()),'auc':float(roc_auc_score(yt,prob))})
    # ground distance derived from empirical differences in fold prevalence, with strictly positive minimum
    prevalence=np.array([z['prevalence'] for z in meta]); D=np.abs(prevalence[:,None]-prevalence[None,:]);D=np.where(D>0,D,0.02)
    D=np.maximum(D,0.02);np.fill_diagonal(D,0.)
    nominal=np.array([z['n_test'] for z in meta],float);nominal/=nominal.sum()
    # fixed weights are illustrative policy priorities, not elicited from clinicians
    w=np.array([.65,.20,.15]); U=np.array([[.10,-.05,-.05],[-.10,.05,.05],[0.,0.,0.]])
    baseline=int(np.argmin(costs.mean(axis=1)@w))
    rows=[]
    for eps in [0.,.0001,.001,.003,.01]:
        ambiguity=DiscreteWasserstein(nominal,D,eps)
        valid=[]
        for k in range(len(thresholds)):
            audit=wasserstein_uniform_gap(costs,k,w,ambiguity)
            if audit['valid']:valid.append(k)
        for k in range(len(thresholds)):
            t0=time.perf_counter()
            exact=exact_stability_radius(costs,k,w,ambiguity,U)
            elapsed=time.perf_counter()-t0
            if exact['radius'] is not None:
                r=exact['radius'];below=audit_radius(costs,k,w,ambiguity,U,max(0,r-1e-5))
                # For r at simplex boundary, validity beyond radius is undefined; don't assert failure.
                bracket=certified_radius_bracket(costs,k,w,ambiguity,U,upper=1.,tol=1e-4)
                gap=wasserstein_uniform_gap(costs,k,w,ambiguity)['gap']
            else:below=None;bracket=None;gap=wasserstein_uniform_gap(costs,k,w,ambiguity)['gap']
            rows.append({'wasserstein_radius':eps,'policy_threshold':float(thresholds[k]),'policy_index':k,
                         'nominal_objectives':costs[k].T@nominal,'uniform_gap':float(gap),
                         'valid_at_center':bool(gap<=1e-7),'stability_radius':exact['radius'],
                         'radius_status':exact['status'],'below_radius_valid':None if below is None else bool(below['valid']),
                         'bisection_status':None if bracket is None else bracket['status'],
                         'bisection_lower':None if bracket is None else bracket.get('lower'),
                         'bisection_upper':None if bracket is None else bracket.get('upper'),
                         'runtime_seconds':elapsed})
    output=ROOT.parent/'Experiments'/'results';output.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    for j,name in enumerate(['false_negative_rate','false_positive_rate','referral_rate']):
        df[name]=df.nominal_objectives.apply(lambda v:float(v[j]))
    df.drop(columns=['nominal_objectives']).to_csv(output/'operational_validation_v12.csv',index=False)
    summary={'dataset':'Wisconsin Diagnostic Breast Cancer (WDBC), scikit-learn bundled copy',
             'dataset_n':len(y),'positive_cases':int(y.sum()),'negative_cases':int((1-y).sum()),
             'policies':thresholds.tolist(),'baseline_policy':float(thresholds[baseline]),
             'folds':meta,'preference_center':w.tolist(),'preference_offset_vertices':U.tolist(),
             'nominal_probabilities':nominal.tolist(),'ground_distance':D.tolist(),
             'wasserstein_radii':[0.,.0001,.001,.003,.01],
             'valid_policy_counts':{str(e):int(df[df.wasserstein_radius==e].valid_at_center.sum()) for e in [0.,.0001,.001,.003,.01]},
             'finite_radius_count':int(df.stability_radius.notna().sum()),
             'radius_below_audit_pass':int((df.below_radius_valid==True).sum()),
             'radius_below_audit_fail':int((df.below_radius_valid==False).sum()),
             'auc_mean':float(np.mean([z['auc'] for z in meta])),
             'scikit_learn_version':__import__('sklearn').__version__, 'python_version':platform.python_version(),
             'limitations':['Real diagnostic observations but hypothetical operational policy; not a clinical recommendation',
             '5 cross-validation folds are constructed scenarios, not independently observed future deployment environments',
             'Ground distance is prevalence-based modeling choice; no empirical transport validation',
             'Thresholds and objective weights are not elicited from stakeholders',
             'No prospective outcomes, cost calibration, external validation or safety review']}
    (output/'operational_validation_v12_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['dataset_n','positive_cases','valid_policy_counts','finite_radius_count','radius_below_audit_pass','radius_below_audit_fail','auc_mean']},indent=2))
    print(df[['wasserstein_radius','policy_threshold','uniform_gap','stability_radius','radius_status']].to_string(index=False))
if __name__=='__main__':run()
