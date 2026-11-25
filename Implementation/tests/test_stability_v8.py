import numpy as np
from algorithms.stability_v8 import audit_radius,certified_radius_bracket
from ambiguity.wasserstein import DiscreteWasserstein

def fixture():
    # target has advantage of 0.2 under w0=(.6,.4), and tie at w=(.5,.5)
    c=np.array([[[0.,1.],[0.,1.]],[[1.,0.],[1.,0.]]])
    amb=DiscreteWasserstein([.5,.5],[[0,1],[1,0]],.2)
    V=np.array([[1.,-1.],[-1.,1.]])
    return c,amb,V

def test_exact_radius():
    c,a,V=fixture();w=[.6,.4]
    r=certified_radius_bracket(c,0,w,a,V,upper=.5,tol=1e-6)
    assert r['status']=='bracketed'
    assert abs(r['lower']-.1)<2e-6
    assert abs(r['upper']-.1)<2e-6

def test_monotone():
    c,a,V=fixture();w=[.6,.4]
    assert audit_radius(c,0,w,a,V,.05)['valid']
    assert not audit_radius(c,0,w,a,V,.15)['valid']

def test_invalid_at_zero():
    c,a,V=fixture()
    assert certified_radius_bracket(c,0,[.4,.6],a,V)['status']=='invalid_at_zero'

def test_bad_vertices():
    c,a,V=fixture()
    import pytest
    with pytest.raises(ValueError):audit_radius(c,0,[.6,.4],a,[[1.,-1.]],.1)
