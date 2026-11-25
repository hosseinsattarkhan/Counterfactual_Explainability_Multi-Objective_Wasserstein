"""Certified (up to numerical tolerance) stability radius for a fixed preference center.
Uncertainty U(t)=w+t conv(V); V must include zero in convex hull.
The exact validity oracle enumerates preference vertices and solves finite-support
Wasserstein LP separation. Bisection returns a certified bracket, not a point proof.
"""
import numpy as np
from scipy.optimize import linprog
from ambiguity.wasserstein import wasserstein_uniform_gap

def _zero_in_hull(V,tol=1e-8):
    k,m=V.shape
    r=linprog(np.zeros(k),A_eq=np.vstack([V.T,np.ones(k)]),
              b_eq=np.r_[np.zeros(m),1.],bounds=(0,None),method='highs')
    return bool(r.success and np.max(np.abs(V.T@r.x))<=tol)

def audit_radius(costs,target,center,ambiguity,vertices,radius,tol=1e-8):
    c=np.asarray(costs,float);w=np.asarray(center,float);V=np.asarray(vertices,float)
    if c.ndim!=3 or V.ndim!=2 or V.shape[1]!=len(w) or c.shape[2]!=len(w):raise ValueError('dimensions')
    if radius<0 or not np.isfinite(radius):raise ValueError('radius')
    if np.max(np.abs(V.sum(axis=1)))>tol or not _zero_in_hull(V):raise ValueError('V must be zero-sum and contain zero in convex hull')
    if abs(w.sum()-1)>tol or np.any(w< -tol):raise ValueError('center not in simplex')
    worst={'gap':-np.inf}
    for j,v in enumerate(V):
        wt=w+radius*v
        if np.min(wt)<-tol:
            return {'valid':False,'gap':float('inf'),'reason':'simplex','vertex':j}
        r=wasserstein_uniform_gap(c,target,wt,ambiguity)
        if r['gap']>worst['gap']:worst={**r,'vertex':j,'reason':'optimality'}
    return worst

def certified_radius_bracket(costs,target,center,ambiguity,vertices,upper=1.,tol=1e-5):
    if upper<=0 or tol<=0:raise ValueError('positive upper and tol required')
    a=audit_radius(costs,target,center,ambiguity,vertices,0.)
    if not a['valid']:return {'status':'invalid_at_zero','lower':None,'upper':0.,'base_gap':a['gap']}
    hi=float(upper); rhi=audit_radius(costs,target,center,ambiguity,vertices,hi)
    if rhi['valid']:return {'status':'valid_through_cap','lower':hi,'upper':None,'at_cap':rhi}
    lo=0.; iterations=0
    while hi-lo>tol:
        mid=(lo+hi)/2
        r=audit_radius(costs,target,center,ambiguity,vertices,mid)
        if r['valid']:lo=mid
        else:hi=mid
        iterations+=1
    return {'status':'bracketed','lower':lo,'upper':hi,'iterations':iterations,
            'valid_at_lower':audit_radius(costs,target,center,ambiguity,vertices,lo),
            'invalid_at_upper':audit_radius(costs,target,center,ambiguity,vertices,hi)}
