import unittest
from pysat.solvers import Solver, Minisat22

class tests_DPLL_Gordon(unittest.TestCase):
    def test_satisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b)
        with Minisat22(bootstrap_with=[[1, 2], [-1, 2], [1, -2]]) as m:
            sat = m.solve()
            self.assertEqual(sat, True)
            # result = m.get_model()
        
    def test_unsatisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b) AND (~a OR ~b)
        with Minisat22(bootstrap_with=[[1, 2], [-1, 2], [1, -2], [-1, -2]]) as m:
            sat = m.solve()
            self.assertEqual(sat, False)
            # result = m.get_model()
        
    def test_satisfiable2(self):
        # (a OR b) AND (~a OR c) AND (b OR ~c)
        with Minisat22(bootstrap_with=[[1, 2], [-1, 3], [2, -3]]) as m:
            sat = m.solve()
            self.assertEqual(sat, True)
            # result = m.get_model()
        
    def test_unsatisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i) AND (a OR ~i)
        with Minisat22(bootstrap_with=[[1, 2, 3, 4, 5, 6, 7, 8],
                                       [9, 10, 11, 12, 13, 14, 15, 16],
                                       [-1, -9], [-2, -10], [-3, -11],
                                       [-4, -12], [-5, -13], [-6, -14],
                                       [-7, -15], [-8, -16], [1, 9],
                                       [-1, 9], [1, -9]]) as m:
            sat = m.solve()
            self.assertEqual(sat, False)
            # result = m.get_model()


        
    def test_satisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i)
        with Minisat22(bootstrap_with=[[1, 2, 3, 4, 5, 6, 7, 8],
                                       [9, 10, 11, 12, 13, 14, 15, 16],
                                       [-1, -9], [-2, -10], [-3, -11],
                                       [-4, -12], [-5, -13], [-6, -14],
                                       [-7, -15], [-8, -16], [1, 9],
                                       [-1, 9]]) as m:
            sat = m.solve()
            self.assertEqual(sat, True)
            # result = m.get_model()
    
if __name__ == '__main__':
    unittest.main()