"""Finite-support 1-Wasserstein ambiguity with a shared distribution.
Transport gamma[i,j] moves nominal mass i to destination j.
All objectives are minimization costs. Exact LP separation on finite support.
"""
import numpy as np
from scipy.optimize import linprog

class DiscreteWasserstein:
    def __init__(self, nominal, ground_distance, radius):
        p=np.asarray(nominal,float); D=np.asarray(ground_distance,float)
        if p.ndim!=1 or len(p)<1 or np.any(p<0) or not np.isfinite(p).all() or abs(p.sum()-1)>1e-8:
            raise ValueError('invalid nominal distribution')
        if D.shape!=(len(p),len(p)) or not np.isfinite(D).all() or np.any(D<0) or np.max(abs(np.diag(D)))>1e-9:
            raise ValueError('invalid ground distance')
        if not np.isfinite(radius) or radius<0:raise ValueError('invalid radius')
        self.p=p;self.D=D;self.radius=float(radius);self.s=len(p)
        self.eq=np.zeros((self.s,self.s*self.s))
        for i in range(self.s):self.eq[i,i*self.s:(i+1)*self.s]=1
        self.ub=self.D.reshape(1,-1)
    def maximize(self, values):
        v=np.asarray(values,float)
        if v.shape!=(self.s,) or not np.isfinite(v).all():raise ValueError('invalid scenario values')
        res=linprog(-np.tile(v,self.s),A_eq=self.eq,b_eq=self.p,
                    A_ub=self.ub,b_ub=[self.radius],bounds=(0,None),method='highs')
        if not res.success:raise RuntimeError(res.message)
        q=res.x.reshape(self.s,self.s).sum(axis=0)
        return {'value':float(-res.fun),'distribution':q,'transport':res.x.reshape(self.s,self.s)}
    def minimize(self,values):
        r=self.maximize(-np.asarray(values,float))
        return {'value':-r['value'],'distribution':r['distribution']}

def wasserstein_uniform_gap(costs,target,weights,ambiguity):
    c=np.asarray(costs,float);w=np.asarray(weights,float)
    if c.ndim!=3 or c.shape[1]!=ambiguity.s or w.shape!=(c.shape[2],):raise ValueError('shape mismatch')
    gaps=[]
    for competitor in range(len(c)):
        vals=(c[target]-c[competitor])@w
        r=ambiguity.maximize(vals)
        gaps.append((r['value'],competitor,r['distribution']))
    val,idx,q=max(gaps,key=lambda x:x[0])
    return {'gap':float(val),'competitor':idx,'adversarial_distribution':q.tolist(),
            'valid':bool(val<=1e-7)}

def wasserstein_worst_case_gap(costs,target,weights,ambiguity):
    c=np.asarray(costs,float);w=np.asarray(weights,float)
    vals=[ambiguity.maximize(c[i]@w)['value'] for i in range(len(c))]
    return {'gap':float(vals[target]-min(vals)),'values':vals,
            'valid':bool(vals[target]-min(vals)<=1e-7)}
