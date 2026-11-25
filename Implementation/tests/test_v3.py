import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from preferences.uncertainty import PreferencePolytope
from counterfactuals.robust_preferences import conservative_preference_counterfactual,preference_robust_gap
from ambiguity.wasserstein import DiscreteWasserstein

def test_preference_ranges():
    p=PreferencePolytope(2,[[1,-1]],[0])
    assert np.allclose(p.ranges(),[[0,.5],[.5,1]])
def test_empty_preferences():
    with pytest.raises(ValueError):PreferencePolytope(2,[[1,0]],[-.1])
def test_robust_zero_radius():
    c=np.array([[[1.,2.],[1.,2.]],[[2.,1.],[2.,1.]]])
    a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
    z=conservative_preference_counterfactual(c,0,[.2,.8],a,0.)
    assert z['status']=='certified_sufficient'
    assert abs(z['l1_distance']-.6)<1e-6
    assert preference_robust_gap(c,0,z['weights'],0,a)['certified']
def test_positive_radius_certificate():
    c=np.array([[[0.,0.],[0.,0.]],[[2.,2.],[2.,2.]]])
    a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.5)
    z=conservative_preference_counterfactual(c,0,[.5,.5],a,.1)
    assert z['status']=='certified_sufficient'
def test_infeasible_robust():
    c=np.array([[[5.,5.],[5.,5.]],[[1.,1.],[1.,1.]]])
    a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.5)
    assert conservative_preference_counterfactual(c,0,[.5,.5],a)['status']=='infeasible'
