import json,time,sys
from pathlib import Path
import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein
from algorithms.final_v6 import cutting_plane,direct_polygon,audit

def run(out):
    rng=np.random.default_rng(20261116);records=[]
    q=np.array([.25,.5,.25]);D=np.abs(np.subtract.outer(np.arange(3),np.arange(3)))
    for n in [4,8,16,32,64]:
        for radius in [0,.1,.3]:
            for replicate in range(4):
                costs=rng.uniform(.2,2.0,(n,3,3))
                # ensure target has some chance of success, but do not select using robust oracle
                costs[0]-=.15
                w0=np.array([.45,.35,.20]);amb=DiscreteWasserstein(q,D,radius)
                offsets=[np.zeros(3),np.array([.02,-.02,0]),np.array([-.02,.02,0])]
                start=time.perf_counter();cg=cutting_plane(costs,0,w0,amb,offsets=offsets);t_cg=time.perf_counter()-start
                start=time.perf_counter();direct=direct_polygon(costs,0,w0,amb,offsets=offsets);t_direct=time.perf_counter()-start
                agree=cg['status']==direct['status'] and (cg['status']!='optimal' or abs(cg['l1_distance']-direct['l1_distance'])<1e-7)
                check=audit(costs,0,cg['weights'],amb,offsets) if cg['status']=='optimal' else None
                records.append({'n':n,'radius':radius,'replicate':replicate,'cg_status':cg['status'],'direct_status':direct['status'],'agreement':bool(agree),'cg_time_s':t_cg,'direct_time_s':t_direct,'cg_cuts':cg['cuts'],'cg_iterations':cg['iterations'],'cg_l1':cg.get('l1_distance'),'direct_l1':direct.get('l1_distance'),'audit':check})
    summary={'cases':len(records),'agreement':sum(r['agreement'] for r in records),'optimal':sum(r['cg_status']=='optimal' for r in records),'infeasible':sum(r['cg_status']=='infeasible' for r in records),'median_cg_s':float(np.median([r['cg_time_s'] for r in records])),'median_direct_s':float(np.median([r['direct_time_s'] for r in records])),'max_l1_difference':float(max([abs(r['cg_l1']-r['direct_l1']) for r in records if r['cg_l1'] is not None],default=0)),'audit_violations':sum(bool(r['audit'] and not r['audit']['valid']) for r in records)}
    Path(out).write_text(json.dumps({'summary':summary,'records':records},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run(sys.argv[1])
