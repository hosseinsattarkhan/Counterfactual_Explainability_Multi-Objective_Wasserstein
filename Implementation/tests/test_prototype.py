import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from counterfactuals.prototype import *
C=np.array([[[0.,8.],[8.,0.],[4.,4.]],[[3.,3.],[3.,3.],[9.,9.]]])
def test_uniform_not_same_as_worst_case():
    w=np.array([.5,.5])
    assert abs(worst_case_gap(C,0,w))<1e-8
    assert uniform_gap(C,0,w)>0

def test_worst_case_counterfactual():
    r=minimum_preference_change(C,0,[.8,.2],'worst_case')
    assert r['status']=='optimal' and r['valid']
    assert abs(r['l1_distance'])<1e-7

def test_uniform_infeasible():
    r=minimum_preference_change(C,0,[.8,.2],'uniform')
    assert r['status']=='infeasible'

def test_target_one_uniform():
    r=minimum_preference_change(C,1,[.8,.2],'worst_case')
    assert r['status']=='infeasible'

def test_preference_constraints():
    a,b=preference_constraints([(1,0)],C)
    assert a.shape==(1,2) and b.shape==(1,)
