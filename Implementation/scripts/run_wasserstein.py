import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ambiguity.wasserstein import DiscreteWasserstein, wasserstein_uniform_gap, wasserstein_worst_case_gap
from counterfactuals.wasserstein_cf import uniform_counterfactual,evaluate_grid
costs=np.array([[[2.,8.],[5.,5.],[8.,2.]],[[6.,3.],[4.,4.],[3.,6.]],[[5.,6.],[3.,7.],[7.,3.]]])
p=np.array([.3,.4,.3]);D=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:]).astype(float)
w0=np.array([.7,.3]);out=[]
for radius in [0.,.1,.3,.6,1.]:
    a=DiscreteWasserstein(p,D,radius)
    for target in range(len(costs)):
        r=uniform_counterfactual(costs,target,w0,a)
        out.append({'radius':radius,'target':target,**r})
path=ROOT.parent/'Experiments'/'results'/'wasserstein_v2.json'
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
