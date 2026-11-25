"""Certified finite-support Wasserstein counterfactuals with fixed preference-offset scenarios.
The learned preference polytope constrains center w; offsets are prescribed stress tests,
not a claim to cover all possible preference errors. Costs: (decision,scenario,objective).
"""
import numpy as np
from scipy.optimize import linprog
from ambiguity.wasserstein import DiscreteWasserstein

def _solve(costs,target,w0,amb,A=None,b=None,offsets=None,full_distributions=None,tol=1e-8,max_iter=300):
    c=np.asarray(costs,float);w0=np.asarray(w0,float);n,s,m=c.shape
    A=np.zeros((0,m)) if A is None else np.asarray(A,float).reshape(-1,m)
    b=np.zeros(len(A)) if b is None else np.asarray(b,float)
    offsets=[np.zeros(m)] if offsets is None else [np.asarray(u,float) for u in offsets]
    if any(abs(u.sum())>1e-9 for u in offsets):raise ValueError('offsets must sum to zero')
    if any(len(u)!=m for u in offsets):raise ValueError('offset dimension')
    if not np.isclose(w0.sum(),1):raise ValueError('w0 must sum to one')
    cuts=[]; rhs=[]; iterations=0
    def add(g,u):
        # E_q[ (w+u) dot (c_target-c_comp)] <= 0
        cuts.append(np.r_[g,np.zeros(m)]);rhs.append(float(-g@u))
    if full_distributions is not None:
        for i in range(n):
            for q in full_distributions:
                g=np.asarray(q)@(c[target]-c[i])
                for u in offsets:add(g,u)
    for k in range(max_iter):
        iterations=k+1
        rows=list(cuts); bounds_rhs=list(rhs)
        for row,limit in zip(A,b):rows.append(np.r_[row,np.zeros(m)]);bounds_rhs.append(limit)
        for u in offsets:
            for j in range(m):
                row=np.zeros(2*m);row[j]=-1;rows.append(row);bounds_rhs.append(u[j])
        for j in range(m):
            row=np.zeros(2*m);row[j]=1;row[m+j]=-1;rows.append(row);bounds_rhs.append(w0[j])
            row=np.zeros(2*m);row[j]=-1;row[m+j]=-1;rows.append(row);bounds_rhs.append(-w0[j])
        r=linprog(np.r_[np.zeros(m),np.ones(m)],A_ub=rows,b_ub=bounds_rhs,
             A_eq=[np.r_[np.ones(m),np.zeros(m)]],b_eq=[1.],
             bounds=[(0,1)]*m+[(0,None)]*m,method='highs')
        if not r.success:
            return {'status':'infeasible' if r.status==2 else 'solver_failure','iterations':iterations,'cuts':len(cuts),'message':r.message}
        w=r.x[:m]
        violations=[]
        for i in range(n):
            for u in offsets:
                gap=(c[target]-c[i])@(w+u)
                sep=amb.maximize(gap)
                if sep['value']>tol:
                    violations.append((sep['value'],i,u,sep['distribution']))
        if full_distributions is not None or not violations:
            return {'status':'optimal','weights':w.tolist(),'l1_distance':float(r.fun),
                    'iterations':iterations,'cuts':len(cuts),'max_violation':float(max([v[0] for v in violations],default=0))}
        worst=max(violations,key=lambda t:t[0]);_,i,u,q=worst
        add(q@(c[target]-c[i]),u)
    return {'status':'iteration_limit','iterations':max_iter,'cuts':len(cuts)}

def cutting_plane(costs,target,w0,amb,A=None,b=None,offsets=None,tol=1e-8):
    return _solve(costs,target,w0,amb,A,b,offsets,tol=tol)

def polygon_vertices_1d3(nominal,radius):
    """Independent polygon clipping: exact W1 on ordered {0,1,2} with distance |i-j|."""
    q=np.asarray(nominal,float)
    if len(q)!=3:raise ValueError('three scenarios required')
    poly=[np.array([0.,0.]),np.array([1.,0.]),np.array([0.,1.])]
    # W1 = |p0-q0|+|p0+p1-(q0+q1)| <= radius
    halfplanes=[]
    for a in (-1,1):
        for b in (-1,1):
            halfplanes.append((np.array([a+b,b],float),radius+a*q[0]+b*(q[0]+q[1])))
    for normal,lim in halfplanes:
        nxt=[]
        for i in range(len(poly)):
            x,y=poly[i],poly[(i+1)%len(poly)]
            fx=float(normal@x-lim);fy=float(normal@y-lim)
            if fx<=1e-10:nxt.append(x)
            if (fx< -1e-10 and fy>1e-10) or (fx>1e-10 and fy< -1e-10):
                nxt.append(x+(y-x)*(-fx)/(fy-fx))
        poly=nxt
        if not poly:break
    out=[]
    for x in poly:
        p=np.array([x[0],x[1],1-x.sum()])
        if not any(np.max(np.abs(p-v))<1e-8 for v in out):out.append(p)
    return out

def direct_polygon(costs,target,w0,amb,A=None,b=None,offsets=None):
    if amb.s!=3 or not np.allclose(amb.D,np.abs(np.subtract.outer(np.arange(3),np.arange(3)))):
        raise ValueError('direct baseline supports ordered 3-scenario line metric')
    return _solve(costs,target,w0,amb,A,b,offsets,full_distributions=polygon_vertices_1d3(amb.p,amb.radius))

def audit(costs,target,w,amb,offsets=None):
    c=np.asarray(costs,float);w=np.asarray(w,float)
    offsets=[np.zeros(len(w))] if offsets is None else offsets
    violations=[]
    for i in range(len(c)):
        for u in offsets:
            sep=amb.maximize((c[target]-c[i])@(w+u))
            violations.append(float(sep['value']))
    return {'max_violation':max(violations),'valid':max(violations)<=1e-7}
