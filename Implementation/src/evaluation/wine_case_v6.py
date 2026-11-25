"""Public UCI Wine dataset via sklearn, with transparent classifier-choice application."""
import json,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from ambiguity.wasserstein import DiscreteWasserstein
from algorithms.final_v6 import cutting_plane,direct_polygon,audit

def run(out):
    data=load_wine();X,y=data.data,data.target
    models=[('dummy',DummyClassifier(strategy='prior')),('logistic',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))),('tree',DecisionTreeClassifier(max_depth=4,random_state=11)),('knn',make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7))),('forest',RandomForestClassifier(n_estimators=40,max_depth=5,random_state=11,n_jobs=1))]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
    costs=[];details=[]
    for name,model in models:
        proba=cross_val_predict(model,X,y,cv=cv,method='predict_proba',n_jobs=1)
        pred=proba.argmax(axis=1);model.fit(X,y)
        size=len(pickle.dumps(model))
        rows=[]
        for k in range(3):
            mask=y==k
            error=np.mean(pred[mask]!=y[mask]);brier=np.mean(np.sum((proba[mask]-np.eye(3)[y[mask]])**2,axis=1))/2
            rows.append([float(error),float(brier),float(size)])
        costs.append(rows);details.append({'model':name,'pickle_bytes':size,'class_error':[r[0] for r in rows]})
    costs=np.asarray(costs,float);sizes=costs[:,:,2].copy();costs[:,:,2]=sizes/np.max(sizes)
    q=np.bincount(y)/len(y);D=np.abs(np.subtract.outer(np.arange(3),np.arange(3)))
    w0=np.array([.6,.35,.05]);offsets=[np.zeros(3),np.array([.02,-.02,0]),np.array([-.02,.02,0])]
    results=[]
    for radius in (0.,.1,.3):
        amb=DiscreteWasserstein(q,D,radius)
        for target in range(len(models)):
            cg=cutting_plane(costs,target,w0,amb,offsets=offsets)
            direct=direct_polygon(costs,target,w0,amb,offsets=offsets)
            results.append({'radius':radius,'target':models[target][0],'cg':cg,'direct':direct,'agreement':cg['status']==direct['status'] and (cg['status']!='optimal' or abs(cg['l1_distance']-direct['l1_distance'])<1e-7),'audit':audit(costs,target,cg['weights'],amb,offsets) if cg['status']=='optimal' else None})
    result={'dataset':'UCI Wine via sklearn.datasets.load_wine','n_samples':len(y),'n_classes':3,'scenario_nominal':q.tolist(),'objectives':['class-conditional misclassification','class-conditional half-Brier score','normalized serialized model size proxy'],'details':details,'results':results,'agreements':sum(x['agreement'] for x in results)}
    Path(out).write_text(json.dumps(result,indent=2));print(json.dumps({'cases':len(results),'agreements':result['agreements'],'optimal':sum(r['cg']['status']=='optimal' for r in results),'infeasible':sum(r['cg']['status']=='infeasible' for r in results)},indent=2))
if __name__=='__main__':run(sys.argv[1])
