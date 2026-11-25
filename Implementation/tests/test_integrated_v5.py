import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ambiguity.wasserstein import DiscreteWasserstein
from counterfactuals.integrated_v5 import learn_preferences,direct_finite_scenario_baseline,exact_joint_audit
class TestV5(unittest.TestCase):
 def setUp(self):
  self.c=np.array([[[1.,3.],[1.,3.]],[[3.,1.],[3.,1.]]]);self.p=np.array([.5,.5]);self.a=DiscreteWasserstein(self.p,np.array([[0,1],[1,0]]),.2)
 def test_learn(self):
  r=learn_preferences(self.c,self.p,[(0,1)]);self.assertEqual(r['status'],'feasible')
 def test_inconsistent(self):
  r=learn_preferences(self.c,self.p,[(0,1),(1,0)]);self.assertEqual(r['status'],'feasible')
 def test_direct(self):
  r=direct_finite_scenario_baseline(self.c,0,np.array([.7,.3]),self.p);self.assertEqual(r['status'],'optimal');self.assertAlmostEqual(r['l1_distance'],0)
 def test_joint(self):
  r=exact_joint_audit(self.c,0,self.a,np.array([[-1.,1.]]),np.array([-.2]));self.assertEqual(r['status'],'certified')
 def test_invalid(self):
  r=exact_joint_audit(self.c,0,self.a,np.array([[1.,-1.]]),np.array([-.2]));self.assertEqual(r['status'],'violated')
if __name__=='__main__':unittest.main()
