"""Exact continuous preference uncertainty via convex-hull vertices.
For each candidate center w, uncertainty is w+conv(offset_vertices).
A shared distribution q belongs to finite-support Wasserstein ball.
"""
import numpy as np
from algorithms.final_v6 import cutting_plane, direct_polygon
from ambiguity.wasserstein import wasserstein_uniform_gap

def solve_continuous_preference(costs, target, w0, ambiguity, offset_vertices, A=None, b=None, method='cutting_plane'):
    vertices=np.asarray(offset_vertices,float)
    w0=np.asarray(w0,float)
    if vertices.ndim!=2 or vertices.shape[1]!=len(w0) or not np.isfinite(vertices).all():
        raise ValueError('invalid preference uncertainty vertices')
    if np.max(np.abs(vertices.sum(axis=1)))>1e-9:
        raise ValueError('offset vertices must preserve unit sum')
    if method=='cutting_plane':
        result=cutting_plane(costs,target,w0,ambiguity,A,b,offsets=vertices)
    elif method=='direct':
        result=direct_polygon(costs,target,w0,ambiguity,A,b,offsets=vertices)
    else:raise ValueError('unknown method')
    result['uncertainty']='continuous convex hull of specified offsets'
    result['exact_for_model']=result['status']=='optimal'
    return result

def audit_continuous(costs,target,center,ambiguity,offset_vertices):
    vertices=np.asarray(offset_vertices,float)
    scores=[wasserstein_uniform_gap(costs,target,np.asarray(center)+u,ambiguity)['gap'] for u in vertices]
    return {'max_violation':float(max(scores)), 'valid':bool(max(scores)<=1e-7),
            'vertex_gaps':[float(v) for v in scores]}
