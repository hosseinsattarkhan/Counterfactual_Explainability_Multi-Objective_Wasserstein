import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein
from counterfactuals.integrated_v4 import preference_vertices,joint_certificate,safe_counterfactual

def test_vertices_simplex():
 v=preference_vertices(np.empty((0,2)),np.array([]),2)
 assert len(v)==2 and np.allclose(v.sum(axis=1),1)
def test_joint_valid():
 c=np.array([[[1,1],[1,1]],[[2,2],[2,2]]],float)
 a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 v=preference_vertices(np.empty((0,2)),[],2)
 assert joint_certificate(c,0,a,v)['status']=='certified'
def test_joint_invalid():
 c=np.array([[[2,2],[2,2]],[[1,1],[1,1]]],float)
 a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 v=preference_vertices(np.empty((0,2)),[],2)
 assert joint_certificate(c,0,a,v)['status']=='violated'
def test_safe_cert():
 c=np.array([[[1,1],[1,1]],[[2,2],[2,2]]],float)
 a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 assert safe_counterfactual(c,0,[.5,.5],a,.1)['status']=='certified_sufficient'
def test_safe_infeasible():
 c=np.array([[[2,2],[2,2]],[[1,1],[1,1]]],float)
 a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 assert safe_counterfactual(c,0,[.5,.5],a,.1)['status']=='infeasible'
