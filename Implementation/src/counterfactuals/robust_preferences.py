"""Joint preference-perturbation and distributional uniform validity.
Find minimal L1 center shift w such that every u in L1 ball around w
(with simplex membership) satisfies all scenario distribution comparisons.
A finite cutting-plane method, with a conservative sufficient certificate.
"""
import numpy as np
from scipy.optimize import linprog
from ambiguity.wasserstein import wasserstein_uniform_gap

def preference_robust_gap(costs,target,center,radius,ambiguity):
    c=np.asarray(costs,float); w=np.asarray(center,float);m=len(w)
    if radius<0:raise ValueError('negative radius')
    # Alternate maximization over u and q is bilinear; cannot certify via alternation.
    # Instead use sound Lipschitz upper bound for any u with ||u-w||_1<=radius:
    # difference in expectation <= radius * max absolute per-objective scenario difference.
    base=wasserstein_uniform_gap(c,target,w,ambiguity)
    bound=float(radius*np.max(np.abs(c[target]-c)))
    return {'nominal_gap':base['gap'],'upper_bound':base['gap']+bound,
            'certified':bool(base['gap']+bound<=1e-7), 'method':'valid_L1_Lipschitz_upper_bound'}

def conservative_preference_counterfactual(costs,target,w0,ambiguity,pref_radius=0.,tol=1e-7,max_iterations=500):
    """Sufficient (not necessary) certificate for all preference deviations.
    Uses scenario-wise max magnitude as conservative Lipschitz constant.
    """
    c=np.asarray(costs,float);w0=np.asarray(w0,float);n,s,m=c.shape
    if pref_radius<0:raise ValueError('negative preference radius')
    lips=np.array([pref_radius*np.max(np.abs(c[target]-c[i])) for i in range(n)])
    rows=[];rhs=[]
    for j in range(m):
        r=np.zeros(2*m);r[j]=1;r[m+j]=-1;rows.append(r);rhs.append(w0[j])
        r=np.zeros(2*m);r[j]=-1;r[m+j]=-1;rows.append(r);rhs.append(-w0[j])
    seen=set()
    for iteration in range(1,max_iterations+1):
        res=linprog(np.r_[np.zeros(m),np.ones(m)],A_ub=rows,b_ub=rhs,
                    A_eq=[np.r_[np.ones(m),np.zeros(m)]],b_eq=[1],
                    bounds=[(0,1)]*m+[(0,None)]*m,method='highs')
        if not res.success:return {'status':'infeasible' if res.status==2 else 'solver_failure','iterations':iteration}
        w=res.x[:m]
        violations=[]
        for i in range(n):
            r=ambiguity.maximize((c[target]-c[i])@w)
            violations.append((r['value']+lips[i],i,r['distribution']))
        gap,i,q=max(violations,key=lambda z:z[0])
        if gap<=tol:return {'status':'certified_sufficient','weights':w.tolist(),
                            'l1_distance':float(np.abs(w-w0).sum()),'upper_bound':float(gap),
                            'iterations':iteration,'pref_radius':pref_radius,
                            'warning':'L1 optimum only within conservative sufficient region'}
        row=np.r_[np.einsum('s,sm->m',q,c[target]-c[i]),np.zeros(m)]
        key=(i,tuple(np.round(row,10)))
        if key in seen:return {'status':'numerical_stall','upper_bound':float(gap)}
        seen.add(key);rows.append(row);rhs.append(-lips[i])
    return {'status':'iteration_limit','iterations':max_iterations}
