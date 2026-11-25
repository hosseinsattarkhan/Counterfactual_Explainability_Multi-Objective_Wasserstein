import numpy as np
from algorithms.exact_radius_v9 import exact_stability_radius
from algorithms.stability_v8 import certified_radius_bracket
from ambiguity.wasserstein import DiscreteWasserstein

def test_known_threshold():
 c=np.array([[[0.,1.],[0.,1.]],[[1.,0.],[1.,0.]]]); a=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
 r=exact_stability_radius(c,0,[.6,.4],a,[[1,-1],[-1,1]])
 assert r['status']=='finite' and abs(r['radius']-.1)<1e-8

def test_random_against_bisection():
 rng=np.random.default_rng(912)
 for z in range(24):
  s=3;m=3;N=5
  c=rng.uniform(.1,2,(N,s,m)); w=np.array([.4,.35,.25]); p=np.array([.2,.3,.5]); D=np.abs(np.arange(s)[:,None]-np.arange(s)[None,:]); a=DiscreteWasserstein(p,D,float(rng.choice([0,.1,.4])))
  # ensure target is uniformly optimal at center by shifting target costs down.
  c[0]-=3
  V=np.array([[.3,-.3,0],[-.3,.3,0],[0,0,0]])
  exact=exact_stability_radius(c,0,w,a,V)
  assert exact['status']=='finite'
  b=certified_radius_bracket(c,0,w,a,V,upper=2.,tol=1e-5)
  assert b['status']=='bracketed'
  assert b['lower']-2e-5 <= exact['radius'] <= b['upper']+2e-5

def test_invalid_center():
 c=np.array([[[1.,0.]],[[0.,1.]]]);a=DiscreteWasserstein([1.],[[0.]],0.)
 r=exact_stability_radius(c,0,[.8,.2],a,[[1,-1],[-1,1]])
 assert r['status']=='invalid_at_zero'
