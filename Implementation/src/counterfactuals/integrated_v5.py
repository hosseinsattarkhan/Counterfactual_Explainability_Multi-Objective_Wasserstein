"""Finite-support exact joint preference/distribution audit and cutting-plane CF.
Weights are centers satisfying learned comparisons; radius r is L1 perturbation around center.
A sufficient center cut is generated using ||delta_s||_infty*r, not an exact joint optimization.
"""
import numpy as np
from scipy.optimize import linprog

def learn_preferences(costs, nominal, comparisons=(), slack_penalty=None):
    c=np.asarray(costs,float);p=np.asarray(nominal,float);m=c.shape[2]
    nominal_cost=np.einsum('s,dsm->dm',p,c)
    A=np.array([nominal_cost[a]-nominal_cost[b] for a,b in comparisons],float).reshape(-1,m)
    if slack_penalty is None:
        r=linprog(np.zeros(m),A_ub=A if len(A) else None,b_ub=np.zeros(len(A)) if len(A) else None,
                  A_eq=np.ones((1,m)),b_eq=[1],bounds=[(0,1)]*m,method='highs')
        return {'status':'feasible' if r.success else 'infeasible','weights':r.x.tolist() if r.success else None,
                'A':A.tolist(),'b':[0.]*len(A)}
    k=len(A); obj=np.r_[np.zeros(m),np.ones(k)*slack_penalty]
    rows=np.hstack((A,-np.eye(k))) if k else None
    r=linprog(obj,A_ub=rows,b_ub=np.zeros(k) if k else None,
              A_eq=[np.r_[np.ones(m),np.zeros(k)]],b_eq=[1],bounds=[(0,1)]*m+[(0,None)]*k,method='highs')
    return {'status':'soft_feasible' if r.success else 'solver_failure','weights':r.x[:m].tolist() if r.success else None,
            'total_slack':float(sum(r.x[m:])) if r.success else None,'A':A.tolist(),'b':[0.]*k}

def exact_joint_audit(costs,target,ambiguity,preference_A,preference_b):
    from counterfactuals.integrated_v4 import preference_vertices, joint_certificate
    m=np.asarray(costs).shape[2]
    V=preference_vertices(preference_A,preference_b,m)
    return joint_certificate(costs,target,ambiguity,V)

def direct_finite_scenario_baseline(costs,target,w0,nominal,comparisons=()):
    """Exact LP for uniform validity at radius 0 (only nominal distribution)."""
    c=np.asarray(costs,float);w0=np.asarray(w0,float);n,s,m=c.shape
    nominal_cost=np.einsum('s,dsm->dm',nominal,c)
    rows=[];rhs=[]
    for i in range(n):rows.append(np.r_[nominal_cost[target]-nominal_cost[i],np.zeros(m)]);rhs.append(0.)
    for a,b in comparisons:rows.append(np.r_[nominal_cost[a]-nominal_cost[b],np.zeros(m)]);rhs.append(0.)
    for j in range(m):
        r=np.zeros(2*m);r[j]=1;r[m+j]=-1;rows.append(r);rhs.append(w0[j])
        r=np.zeros(2*m);r[j]=-1;r[m+j]=-1;rows.append(r);rhs.append(-w0[j])
    res=linprog(np.r_[np.zeros(m),np.ones(m)],A_ub=rows,b_ub=rhs,
      A_eq=[np.r_[np.ones(m),np.zeros(m)]],b_eq=[1.],bounds=[(0,1)]*m+[(0,None)]*m,method='highs')
    return {'status':'optimal' if res.success else 'infeasible' if res.status==2 else 'solver_failure',
            'weights':res.x[:m].tolist() if res.success else None,
            'l1_distance':float(res.fun) if res.success else None}
