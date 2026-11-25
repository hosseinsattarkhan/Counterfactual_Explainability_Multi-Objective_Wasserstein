import json,time
from pathlib import Path
import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein
from algorithms.exact_radius_v9 import exact_stability_radius
from algorithms.stability_v8 import certified_radius_bracket

def run(out):
 rng=np.random.default_rng(20261116);rows=[]
 for n in [4,8,16,32]:
  for eps in [0.,.1,.3]:
   for seed in range(3):
    s=4;m=3;c=rng.uniform(.2,2.,(n,s,m));c[0]-=2.5
    w=np.array([.45,.35,.20]);V=np.array([[.25,-.25,0],[-.25,.25,0],[0,0,0]])
    a=DiscreteWasserstein([.25]*4,np.abs(np.subtract.outer(np.arange(4),np.arange(4))),eps)
    t=time.perf_counter();exact=exact_stability_radius(c,0,w,a,V); te=time.perf_counter()-t
    t=time.perf_counter();bis=certified_radius_bracket(c,0,w,a,V,upper=3.,tol=1e-5);tb=time.perf_counter()-t
    rows.append({'n':n,'epsilon':eps,'replicate':seed,'exact_radius':exact['radius'],
                 'exact_status':exact['status'],'bisection_status':bis['status'],
                 'bracket_lower':bis.get('lower'),'bracket_upper':bis.get('upper'),
                 'exact_seconds':te,'bisection_seconds':tb,
                 'agree':bis['status']=='bracketed' and bis['lower']-2e-5<=exact['radius']<=bis['upper']+2e-5})
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'rows':rows},indent=2))
 print(json.dumps({'cases':len(rows),'agreements':sum(r['agree'] for r in rows),
                   'median_exact_seconds':float(np.median([r['exact_seconds'] for r in rows])),
                   'median_bisection_seconds':float(np.median([r['bisection_seconds'] for r in rows]))},indent=2))
if __name__=='__main__':run('Experiments/results/radius_benchmark_v9.json')
