"""Preference inference from pairwise decisions and uncertainty via LP."""
import numpy as np
from scipy.optimize import linprog

class PreferencePolytope:
    def __init__(self, dimension, inequalities=None, rhs=None, lower=0.0):
        self.m=dimension
        self.A=np.asarray(inequalities if inequalities is not None else np.empty((0,dimension)),float).reshape(-1,dimension)
        self.b=np.asarray(rhs if rhs is not None else np.empty(0),float)
        self.lower=float(lower)
        if self.A.shape[0]!=len(self.b) or dimension<2 or lower<0 or lower*dimension>1:raise ValueError('invalid preference constraints')
        if not self.optimize(np.zeros(dimension))['success']:raise ValueError('empty preference set')
    @classmethod
    def from_comparisons(cls,nominal_costs,comparisons,margin=0.,lower=0.):
        c=np.asarray(nominal_costs,float)
        # a preferred to b means (c[a]-c[b]) @ w <= -margin
        A=[c[a]-c[b] for a,b in comparisons]
        return cls(c.shape[1],A,[-margin]*len(A),lower)
    def optimize(self, direction, maximize=False):
        direction=np.asarray(direction,float)
        if direction.shape!=(self.m,):raise ValueError('direction shape')
        res=linprog(-direction if maximize else direction,A_ub=self.A if len(self.A) else None,
                    b_ub=self.b if len(self.b) else None,A_eq=np.ones((1,self.m)),b_eq=[1],
                    bounds=[(self.lower,1)]*self.m,method='highs')
        return {'success':bool(res.success),'weights':res.x.tolist() if res.success else None,
                'value':float(direction@res.x) if res.success else None}
    def ranges(self):
        return [[self.optimize(np.eye(self.m)[j])['value'],self.optimize(np.eye(self.m)[j],True)['value']] for j in range(self.m)]
