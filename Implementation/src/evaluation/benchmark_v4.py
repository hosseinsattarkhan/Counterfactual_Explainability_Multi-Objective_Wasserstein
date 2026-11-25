"""Reproducible synthetic facility-choice case: no claim of real-world dataset."""
import json,time,sys
from pathlib import Path
import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein,wasserstein_uniform_gap
from counterfactuals.wasserstein_cf import uniform_counterfactual
from counterfactuals.integrated_v4 import safe_counterfactual,joint_certificate,preference_vertices

def run():
    rng=np.random.default_rng(20261116);out=[]
    for n in (4,8,16,32):
      for radius in (0.,.1,.3):
       for seed in range(3):
        # Synthetic cost, delay and emissions of n hypothetical facility designs.
        c=rng.uniform(1,10,(n,4,3)); c+=np.linspace(0,2,n)[:,None,None]*np.array([1,-.7,.4])[None,None,:]
        amb=DiscreteWasserstein(np.ones(4)/4,np.abs(np.arange(4)[:,None]-np.arange(4)[None,:]),radius)
        w0=np.array([.5,.3,.2]); target=int(np.argmin(np.einsum('s,dsm,m->d',amb.p,c,w0)))
        start=time.perf_counter();base=uniform_counterfactual(c,target,w0,amb);tbase=time.perf_counter()-start
        start=time.perf_counter();safe=safe_counterfactual(c,target,w0,amb,.04);tsafe=time.perf_counter()-start
        # Independent fixed-polytope joint audit around the returned safe center.
        audit=None
        if safe['status']=='certified_sufficient':
         w=np.array(safe['weights']);eps=.02
         A=np.vstack((np.eye(3),-np.eye(3)));b=np.r_[w+eps,-w+eps]
         verts=preference_vertices(A,b,3)
         audit=joint_certificate(c,target,amb,verts)
        out.append({'n':n,'radius':radius,'replicate':seed,'target':target,'base_status':base['status'],
          'base_seconds':round(tbase,6),'base_iterations':base.get('iterations'),
          'safe_status':safe['status'],'safe_seconds':round(tsafe,6),'safe_iterations':safe.get('iterations'),
          'safe_l1':safe.get('l1_distance'),'independent_vertex_audit':audit})
    return out
if __name__=='__main__':
    result=run();p=Path(__file__).resolve().parents[3]/'Experiments'/'results'/'benchmark_v4.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2));
    print(json.dumps({'runs':len(result),'base_certified':sum(r['base_status']=='certified' for r in result),'safe_certified':sum(r['safe_status']=='certified_sufficient' for r in result),'audit_violations':sum(r['independent_vertex_audit'] is not None and r['independent_vertex_audit']['status']=='violated' for r in result)},indent=2))
