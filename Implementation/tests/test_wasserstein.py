import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein,wasserstein_uniform_gap
from counterfactuals.wasserstein_cf import uniform_counterfactual

def test_zero_radius():
 a=DiscreteWasserstein([.25,.75],[[0,1],[1,0]],0)
 r=a.maximize([2,6]);assert abs(r['value']-5)<1e-8

def test_full_radius():
 a=DiscreteWasserstein([.25,.75],[[0,1],[1,0]],1)
 assert abs(a.maximize([2,6])['value']-6)<1e-8

def test_monotonic_radius():
 v=[DiscreteWasserstein([.5,.5],[[0,1],[1,0]],e).maximize([1,3])['value'] for e in [0,.2,.5,1]]
 assert np.all(np.diff(v)>=-1e-9)

def test_counterfactual_exact():
 c=np.array([[[0.,2.],[0.,2.]],[[2.,0.],[2.,0.]]]);a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.4)
 r=uniform_counterfactual(c,1,[.8,.2],a)
 assert r['status']=='certified' and abs(r['l1_distance']-.6)<1e-7
 assert wasserstein_uniform_gap(c,1,r['weights'],a)['valid']

def test_infeasible():
 c=np.array([[[0.,0.],[0.,0.]],[[1.,1.],[1.,1.]]]);a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 assert uniform_counterfactual(c,1,[.5,.5],a)['status']=='infeasible'

def test_robust_gap_separation():
 c=np.array([[[0.,3.],[3.,0.]],[[2.,1.],[1.,2.]]]);a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.5)
 r=wasserstein_uniform_gap(c,0,[.5,.5],a)
 assert r['gap']>=-1e-8
