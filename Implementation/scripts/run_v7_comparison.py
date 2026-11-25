import sys,json,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from ambiguity.wasserstein import DiscreteWasserstein
from algorithms.continuous_preference_v7 import solve_continuous_preference,audit_continuous
rng=np.random.default_rng(20261116)
rows=[]
for n in (4,8,16):
 for radius in (0.,.1,.3):
  for seed in range(3):
   costs=rng.uniform(.1,10,size=(n,3,2));w0=np.array([.5,.5]);target=int(np.argmin(costs.mean(axis=1)@w0))
   amb=DiscreteWasserstein([1/3]*3,np.abs(np.subtract.outer(np.arange(3),np.arange(3))),radius)
   u=[[.03,-.03],[-.03,.03]]
   results={}
   for method in ('cutting_plane','direct'):
    t=time.perf_counter();r=solve_continuous_preference(costs,target,w0,amb,u,method=method)
    results[method]={'status':r['status'],'objective':r.get('l1_distance'),'seconds':time.perf_counter()-t}
    if r['status']=='optimal':results[method]['audit']=audit_continuous(costs,target,r['weights'],amb,u)['valid']
   rows.append({'n':n,'radius':radius,'replicate':seed,**results})
summary={'instances':len(rows),'status_agreements':sum(x['direct']['status']==x['cutting_plane']['status'] for x in rows),
'objective_agreements':sum(abs(x['direct']['objective']-x['cutting_plane']['objective'])<1e-7 for x in rows if x['direct']['status']=='optimal' and x['cutting_plane']['status']=='optimal'),
'both_optimal':sum(x['direct']['status']=='optimal' and x['cutting_plane']['status']=='optimal' for x in rows),
'failed_audits':sum(x[k].get('audit') is False for x in rows for k in ('direct','cutting_plane')),
'median_seconds_cutting_plane':float(np.median([x['cutting_plane']['seconds'] for x in rows])),
'median_seconds_direct':float(np.median([x['direct']['seconds'] for x in rows]))}
out=root.parent/'Experiments'/'results'/'continuous_preferences_v7.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
