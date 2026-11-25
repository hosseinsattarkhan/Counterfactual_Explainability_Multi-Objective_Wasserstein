import numpy as np
from algorithms.final_v6 import cutting_plane,direct_polygon,audit,polygon_vertices_1d3
from ambiguity.wasserstein import DiscreteWasserstein

def test_polygon_support():
    q=np.array([.2,.5,.3]);a=DiscreteWasserstein(q,np.abs(np.subtract.outer(np.arange(3),np.arange(3))),.2)
    vertices=polygon_vertices_1d3(q,.2)
    for v in np.eye(3):
        assert abs(max(v@p for p in vertices)-a.maximize(v)['value'])<1e-8

def test_exact_agreement():
    rng=np.random.default_rng(33)
    for radius in (0.,.12,.35):
        for _ in range(5):
            c=rng.uniform(.1,2,(5,3,3));w0=np.array([.5,.3,.2]);q=np.array([.2,.5,.3]);D=np.abs(np.subtract.outer(np.arange(3),np.arange(3)))
            a=DiscreteWasserstein(q,D,radius)
            offsets=[np.zeros(3),np.array([.02,-.02,0])]
            x=cutting_plane(c,0,w0,a,offsets=offsets)
            y=direct_polygon(c,0,w0,a,offsets=offsets)
            assert x['status']==y['status'],(x,y)
            if x['status']=='optimal':
                assert abs(x['l1_distance']-y['l1_distance'])<1e-7
                assert audit(c,0,x['weights'],a,offsets)['valid']
