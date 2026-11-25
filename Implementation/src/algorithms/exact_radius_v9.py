"""Exact critical preference radius via linear-fractional transport LP.

For each competitor and offset vertex, find first t where a(pi)+t*b(pi)>0.
The infimum of -a(pi)/b(pi) over b(pi)>0 is computed by Charnes-Cooper.
Requires base uniform validity; numerical tolerances are reported separately.
"""
import numpy as np
from scipy.optimize import linprog
from algorithms.stability_v8 import audit_radius, _zero_in_hull


def exact_stability_radius(costs, target, center, ambiguity, vertices, tol=1e-8):
    c=np.asarray(costs,float); w=np.asarray(center,float); V=np.asarray(vertices,float)
    if c.ndim!=3 or c.shape[1]!=ambiguity.s or c.shape[2]!=len(w): raise ValueError('cost dimensions')
    if V.ndim!=2 or V.shape[1]!=len(w) or not np.isfinite(V).all(): raise ValueError('vertices')
    if abs(w.sum()-1)>tol or np.min(w)<-tol or np.max(np.abs(V.sum(axis=1)))>tol or not _zero_in_hull(V): raise ValueError('invalid simplex or offsets')
    if not audit_radius(c,target,w,ambiguity,V,0,tol=tol)['valid']:
        return {'status':'invalid_at_zero','radius':None}
    # Simplex feasibility cap. Zero directional coordinates do not constrain.
    cap=float('inf'); cap_witness=None
    for j,v in enumerate(V):
        for k,d in enumerate(v):
            if d < -tol:
                r=-w[k]/d
                if r < cap: cap=float(r);cap_witness={'vertex':j,'coordinate':k}
    s=ambiguity.s; n=s*s; p=ambiguity.p; D=ambiguity.D.reshape(-1)
    Aeq=np.zeros((s+1,n+1))
    Aeq[:s,:n]=ambiguity.eq
    Aeq[:s,n]=-p
    Aub=np.zeros((1,n+1)); Aub[0,:n]=D; Aub[0,n]=-ambiguity.radius
    best=cap; witness={'type':'simplex','details':cap_witness} if np.isfinite(cap) else None
    for i in range(len(c)):
        if i==target:continue
        delta=c[target]-c[i]
        for j,v in enumerate(V):
            a=np.tile(delta@w,s); b=np.tile(delta@v,s)
            if np.max(b)<=tol:continue
            # z=pi/(b*pi), h=1/(b*pi); b*z=1; objective -a*z.
            Aeq[s,:n]=b
            beq=np.r_[np.zeros(s),1.]
            res=linprog(np.r_[-a,0.],A_eq=Aeq,b_eq=beq,A_ub=Aub,b_ub=[0.],bounds=(0,None),method='highs')
            if res.status==2:continue
            if not res.success:raise RuntimeError('fractional LP: '+res.message)
            radius=float(res.fun)
            if radius < best:
                best=max(0.,radius)
                witness={'type':'optimality','competitor':i,'vertex':j,
                         'transport':(res.x[:n]/res.x[n]).reshape(s,s).tolist() if res.x[n]>1e-12 else None,
                         'fractional_lp_objective':radius}
    return {'status':'finite' if np.isfinite(best) else 'unbounded',
            'radius':float(best) if np.isfinite(best) else None,
            'limiting_constraint':witness,
            'note':'Analytical threshold in exact arithmetic; LP numerical tolerances apply. Ties count as valid.'}
