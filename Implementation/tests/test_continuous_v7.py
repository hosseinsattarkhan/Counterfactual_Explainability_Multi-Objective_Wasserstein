import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from ambiguity.wasserstein import DiscreteWasserstein
from algorithms.continuous_preference_v7 import solve_continuous_preference,audit_continuous

def instance():
    c=np.array([[[2.,5.],[3.,4.],[4.,3.]],[[4.,2.],[4.,3.],[3.,4.]]])
    amb=DiscreteWasserstein([.4,.3,.3],np.abs(np.subtract.outer(np.arange(3),np.arange(3))),.2)
    return c,amb

def test_direct_matches_generation():
    c,amb=instance();u=np.array([[.02,-.02],[-.02,.02]])
    a=solve_continuous_preference(c,0,[.5,.5],amb,u,method='direct')
    b=solve_continuous_preference(c,0,[.5,.5],amb,u,method='cutting_plane')
    assert a['status']==b['status']
    if a['status']=='optimal':
        assert abs(a['l1_distance']-b['l1_distance'])<1e-7
        assert audit_continuous(c,0,b['weights'],amb,u)['valid']

def test_singleton_uncertainty():
    c,amb=instance();r=solve_continuous_preference(c,0,[.5,.5],amb,[[0,0]])
    assert r['status'] in ('optimal','infeasible')

def test_reject_nonzero_sum():
    c,amb=instance()
    with pytest.raises(ValueError):solve_continuous_preference(c,0,[.5,.5],amb,[[.1,.1]])
