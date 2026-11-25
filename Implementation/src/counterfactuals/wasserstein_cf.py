"""Exact finite-support Wasserstein uniform counterfactual via LP cutting planes.
Each separation LP maximizes the target-competitor cost difference over
transport plans. Feasibility is certified up to explicit tolerance.
"""
import numpy as np
from scipy.optimize import linprog
from ambiguity.wasserstein import wasserstein_uniform_gap, wasserstein_worst_case_gap

def uniform_counterfactual(costs,target,w0,ambiguity,comparisons=(),tol=1e-7,max_iterations=1000):
    c=np.asarray(costs,float);w0=np.asarray(w0,float)
    if c.ndim!=3 or c.shape[1]!=ambiguity.s or w0.shape!=(c.shape[2],):raise ValueError('shape mismatch')
    n,s,m=c.shape
    if target<0 or target>=n or np.any(w0<0) or abs(w0.sum()-1)>1e-8:raise ValueError('invalid target or weights')
    obj=np.r_[np.zeros(m),np.ones(m)]; rows=[];rhs=[]
    for j in range(m):
        r=np.zeros(2*m);r[j]=1;r[m+j]=-1;rows.append(r);rhs.append(w0[j])
        r=np.zeros(2*m);r[j]=-1;r[m+j]=-1;rows.append(r);rhs.append(-w0[j])
    nominal_cost=np.einsum('s,dsm->dm',ambiguity.p,c)
    for a,b in comparisons:
        if not(0<=a<n and 0<=b<n):raise ValueError('invalid comparison')
        rows.append(np.r_[nominal_cost[a]-nominal_cost[b],np.zeros(m)]);rhs.append(0.)
    seen=set()
    for iteration in range(1,max_iterations+1):
        res=linprog(obj,A_ub=np.asarray(rows),b_ub=np.asarray(rhs),
            A_eq=[np.r_[np.ones(m),np.zeros(m)]],b_eq=[1],
            bounds=[(0,1)]*m+[(0,None)]*m,method='highs')
        if not res.success:return {'status':'infeasible' if res.status==2 else 'solver_failure',
                                   'iterations':iteration,'message':res.message}
        w=res.x[:m];sep=wasserstein_uniform_gap(c,target,w,ambiguity)
        if sep['gap']<=tol:
            return {'status':'certified','weights':w.tolist(),
                    'l1_distance':float(np.abs(w-w0).sum()),'certificate_gap':sep['gap'],
                    'iterations':iteration,'tolerance':tol,'target':target,
                    'certificate_type':'uniform_for_all_distributions_in_finite_support_W1_ball'}
        i=sep['competitor'];q=np.asarray(sep['adversarial_distribution'])
        row=np.r_[np.einsum('s,sm->m',q,c[target]-c[i]),np.zeros(m)]
        key=tuple(np.round(row,11))
        if key in seen:return {'status':'numerical_stall','gap':sep['gap'],'iterations':iteration}
        seen.add(key);rows.append(row);rhs.append(0.)
    return {'status':'iteration_limit','iterations':max_iterations}

def evaluate_grid(costs,target,w0,ambiguity,steps=51):
    """Diagnostic ONLY for 2 objectives, not an optimality certificate."""
    if len(w0)!=2:raise ValueError('grid requires 2 objectives')
    rows=[]
    for v in np.linspace(0,1,steps):
        w=np.array([v,1-v]);u=wasserstein_uniform_gap(costs,target,w,ambiguity)
        z=wasserstein_worst_case_gap(costs,target,w,ambiguity)
        rows.append({'w1':float(v),'uniform_gap':u['gap'],'worst_case_gap':z['gap'],
                     'l1_distance':float(np.abs(w-np.asarray(w0)).sum())})
    return rows
