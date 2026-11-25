"""Joint preference/distribution certificates, exact finite vertex audit, and safe robust CF.
All claims concern finite decisions, finite scenario support, and a shared W1 distribution.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from ambiguity.wasserstein import wasserstein_uniform_gap
from counterfactuals.wasserstein_cf import uniform_counterfactual

def preference_vertices(A,b,m,tol=1e-8):
    """Enumerate vertices of {w>=0, sum(w)=1, A w<=b}; small m only."""
    A=np.asarray(A,float).reshape(-1,m);b=np.asarray(b,float).reshape(-1)
    G=np.vstack((A,-np.eye(m)));h=np.r_[b,np.zeros(m)]
    vertices=[]
    for ix in itertools.combinations(range(len(G)),m-1):
        M=np.vstack((np.ones(m),G[list(ix)]));rhs=np.r_[1.,h[list(ix)]]
        if np.linalg.matrix_rank(M)<m:continue
        w=np.linalg.solve(M,rhs)
        if np.all(G@w<=h+tol) and not any(np.linalg.norm(w-v,np.inf)<tol for v in vertices):vertices.append(w)
    return np.asarray(vertices).reshape(-1,m)

def joint_certificate(costs,target,ambiguity,vertices,tol=1e-7):
    """Exact worst gap over every preference in a fixed polytope and every P in W1 ball.
    Vertex sufficiency follows from convexity of supremum of linear functions in w.
    """
    V=np.asarray(vertices,float)
    if len(V)==0:return {'status':'empty_preference_set'}
    outcomes=[wasserstein_uniform_gap(costs,target,w,ambiguity) for w in V]
    worst=max(range(len(V)),key=lambda i:outcomes[i]['gap'])
    return {'status':'certified' if outcomes[worst]['gap']<=tol else 'violated',
            'gap':float(outcomes[worst]['gap']),'worst_vertex':V[worst].tolist(),
            'competitor':outcomes[worst]['competitor'],'vertices_checked':len(V)}

def safe_counterfactual(costs,target,w0,ambiguity,preference_radius=0.,comparisons=(),tol=1e-7):
    """Sufficient certificate for all w in simplex with ||w-wcenter||_1<=radius.
    Bound gap(w) <= gap(center)+radius*max_s ||f_target(s)-f_i(s)||_inf.
    Adds a safe RHS margin for each competitor (possibly conservative).
    """
    if preference_radius<0:raise ValueError('negative radius')
    c=np.asarray(costs,float);w0=np.asarray(w0,float);n,s,m=c.shape
    obj=np.r_[np.zeros(m),np.ones(m)];rows=[];rhs=[]
    for j in range(m):
        r=np.zeros(2*m);r[j]=1;r[m+j]=-1;rows.append(r);rhs.append(w0[j])
        r=np.zeros(2*m);r[j]=-1;r[m+j]=-1;rows.append(r);rhs.append(-w0[j])
    nominal=np.einsum('s,dsm->dm',ambiguity.p,c)
    for a,b in comparisons:rows.append(np.r_[nominal[a]-nominal[b],np.zeros(m)]);rhs.append(0.)
    margins=np.max(np.abs(c[target]-c),axis=(1,2))*preference_radius
    for it in range(1,1001):
        res=linprog(obj,A_ub=rows,b_ub=rhs,A_eq=[np.r_[np.ones(m),np.zeros(m)]],b_eq=[1.],bounds=[(0,1)]*m+[(0,None)]*m,method='highs')
        if not res.success:return {'status':'infeasible' if res.status==2 else 'solver_failure','iterations':it}
        w=res.x[:m]
        viol=[]
        for i in range(n):
            sep=ambiguity.maximize((c[target]-c[i])@w)
            viol.append((sep['value']+margins[i],i,sep['distribution']))
        gap,i,q=max(viol,key=lambda t:t[0])
        if gap<=tol:return {'status':'certified_sufficient','weights':w.tolist(),'l1_distance':float(np.abs(w-w0).sum()),'safe_gap':float(gap),'iterations':it,'radius':preference_radius}
        row=np.r_[np.einsum('s,sm->m',q,c[target]-c[i]),np.zeros(m)]
        rows.append(row);rhs.append(-margins[i])
    return {'status':'iteration_limit'}
