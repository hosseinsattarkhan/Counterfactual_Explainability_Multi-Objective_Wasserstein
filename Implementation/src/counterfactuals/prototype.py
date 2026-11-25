"""Finite-decision, finite-scenario, shared-distribution counterfactual prototype.
All objectives are costs. Uncertainty is a probability simplex over scenarios.
Uniform certificate is exact for this ambiguity set; worst-case is a minimax
selection (NOT the sum of objective-wise worst cases).
"""
import numpy as np
from scipy.optimize import linprog

TOL = 1e-8


def check_costs(costs):
    c = np.asarray(costs, dtype=float)
    if c.ndim != 3 or min(c.shape) < 1 or not np.isfinite(c).all():
        raise ValueError('costs must have shape (decisions, scenarios, objectives)')
    return c


def preference_constraints(comparisons, costs, margin=0.0):
    """For observed (preferred, rejected) pairs, enforce nominal expected
    cost(preferred) <= cost(rejected) - margin, using uniform scenario mean.
    Returns A_ub, b_ub for scipy linprog; empty comparisons are supported.
    """
    c=check_costs(costs); m=c.shape[2]; rows=[]; rhs=[]
    means=c.mean(axis=1)
    for preferred,rejected in comparisons:
        if not (0<=preferred<c.shape[0] and 0<=rejected<c.shape[0]):
            raise ValueError('invalid decision index')
        rows.append(means[preferred]-means[rejected]);rhs.append(-float(margin))
    return np.asarray(rows,dtype=float).reshape(-1,m),np.asarray(rhs,dtype=float)


def robust_value(costs, decision, w):
    c=check_costs(costs);w=np.asarray(w,float)
    return float(np.max(c[decision]@w))


def uniform_gap(costs,target,w):
    """Max over competitors and scenarios of target minus competitor cost.
    Nonpositive iff target is optimal for EVERY distribution in simplex.
    """
    c=check_costs(costs);w=np.asarray(w,float)
    return float(np.max(np.einsum('dsm,m->ds',c[target][None,:,:]-c,w)))


def worst_case_gap(costs,target,w):
    c=check_costs(costs)
    vals=np.max(c@np.asarray(w,float),axis=1)
    return float(vals[target]-np.min(vals))


def minimum_preference_change(costs,target,w0,mode='uniform',comparisons=(),margin=0.0):
    """Exact L1 minimum perturbation for uniform validity. For worst-case,
    finite scenario selector enumeration yields a global optimum for simplex
    ambiguity; exponential in number of decisions (and scenarios). Intended
    only for small verification instances. Tie-inclusive optimality.
    """
    c=check_costs(costs);n,s,m=c.shape
    w0=np.asarray(w0,float)
    if w0.shape!=(m,) or (w0 < -TOL).any() or abs(w0.sum()-1)>1e-7:
        raise ValueError('w0 must lie on the weight simplex')
    if not 0<=target<n:raise ValueError('invalid target')
    A,b=preference_constraints(comparisons,c,margin)
    # variables [w (m), t (m)]
    obj=np.r_[np.zeros(m),np.ones(m)]
    baseline=[]; bounds=[(0,1)]*m+[(0,None)]*m
    for j in range(m):
        r=np.zeros(2*m);r[j]=1;r[m+j]=-1;baseline.append((r,w0[j]))
        r=np.zeros(2*m);r[j]=-1;r[m+j]=-1;baseline.append((r,-w0[j]))
    def solve(extra):
        rows=[r for r,_ in baseline]; rhs=[v for _,v in baseline]
        for row,v in zip(A,b):rows.append(np.r_[row,np.zeros(m)]);rhs.append(v)
        for row,v in extra:rows.append(np.r_[row,np.zeros(m)]);rhs.append(v)
        res=linprog(obj,A_ub=np.array(rows),b_ub=np.array(rhs),
                    A_eq=np.array([np.r_[np.ones(m),np.zeros(m)]]),b_eq=[1.],
                    bounds=bounds,method='highs')
        return res
    if mode=='uniform':
        extra=[(c[target,k]-c[i,k],0.) for i in range(n) for k in range(s)]
        best=solve(extra)
    elif mode=='worst_case':
        # For each decision, pick one scenario attaining its maximum.
        # Enforce selected scenario dominates all other scenarios, then
        # compare target's selected maximum to each competitor's maximum.
        import itertools
        if s**n>100000:raise ValueError('enumeration too large for V1')
        best=None
        for active in itertools.product(range(s),repeat=n):
            extra=[]
            for i in range(n):
                extra.extend((c[i,k]-c[i,active[i]],0.) for k in range(s))
                extra.append((c[target,active[target]]-c[i,active[i]],0.))
            r=solve(extra)
            if r.success and (best is None or r.fun<best.fun-1e-9):best=r
    else:raise ValueError('mode must be uniform or worst_case')
    if best is None or not best.success:
        return {'status':'infeasible','mode':mode,'target':target}
    w=best.x[:m]
    gap=uniform_gap(c,target,w) if mode=='uniform' else worst_case_gap(c,target,w)
    return {'status':'optimal','mode':mode,'target':target,
            'weights':w.tolist(),'l1_distance':float(np.abs(w-w0).sum()),
            'certificate_gap':gap,'valid':bool(gap<=1e-7)}
