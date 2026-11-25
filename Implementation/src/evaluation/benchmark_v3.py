"""Reproducible controlled benchmark. All costs synthetic, no external data."""
import json, time
import numpy as np
from pathlib import Path
from ambiguity.wasserstein import DiscreteWasserstein, wasserstein_uniform_gap, wasserstein_worst_case_gap
from counterfactuals.wasserstein_cf import uniform_counterfactual
from counterfactuals.robust_preferences import conservative_preference_counterfactual,preference_robust_gap
from preferences.uncertainty import PreferencePolytope

def run(output):
    rng=np.random.default_rng(20261116)
    rows=[]
    for n in [3,6,12,24]:
      for seed in range(3):
        # opposing objectives plus scenario shocks, all synthetic
        r=np.random.default_rng(1000*n+seed)
        costs=r.uniform(1,8,(n,4,3)); costs[:,:,1]=10-costs[:,:,0]+r.normal(0,.3,(n,4))
        costs=np.maximum(costs,.01)
        D=np.abs(np.subtract.outer(np.arange(4),np.arange(4))).astype(float)
        w0=np.array([.6,.25,.15]); target=n//2
        for eps in [0.,.2,.6]:
          ambiguity=DiscreteWasserstein(np.ones(4)/4,D,eps)
          start=time.perf_counter(); sol=uniform_counterfactual(costs,target,w0,ambiguity,max_iterations=300)
          elapsed=time.perf_counter()-start
          record={'n':n,'seed':seed,'radius':eps,'status':sol['status'],
                  'l1':sol.get('l1_distance'),'gap':sol.get('certificate_gap'),
                  'iterations':sol.get('iterations'),'runtime_s':round(elapsed,6)}
          if sol['status']=='certified':
            check=wasserstein_uniform_gap(costs,target,sol['weights'],ambiguity)
            assert check['gap']<=1e-6
            record['independent_gap']=check['gap']
          rows.append(record)
    # Controlled explicit difference between uniform and WC optimality
    special=np.array([[[3.,1.],[3.,1.]],[[2.,2.],[2.,2.]]])
    amb=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.5)
    discrepancy=[]
    for w1 in np.linspace(0,1,11):
        w=[float(w1),float(1-w1)]
        discrepancy.append({'w1':w1,'uniform_gap':wasserstein_uniform_gap(special,0,w,amb)['gap'],
                            'worst_case_gap':wasserstein_worst_case_gap(special,0,w,amb)['gap']})
    # preference polytope with observed preference for decision 0 over 1
    nominal=np.array([[2.,4.,5.],[4.,3.,4.],[5.,5.,2.]])
    pref=PreferencePolytope.from_comparisons(nominal,[(0,1)])
    result={'metadata':{'dataset':'synthetic','seed':20261116,'n_runs':len(rows),
                        'note':'Timing is machine-specific; no statistical superiority claim'},
            'runs':rows,'discrepancy':discrepancy,'preference_ranges':pref.ranges()}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2))
    return result
