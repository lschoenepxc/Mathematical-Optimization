from DPLL_Gordon import dpll
import unittest

class tests_DPLL_Gordon(unittest.TestCase):
    def test_satisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b)
        ex_sat1 = ['and',
                    ['or', 'a', 'b'],
                    ['or', ['not', 'a'], 'b'],
                    ['or', 'a', ['not', 'b']]]
        print(dpll(ex_sat1))
        
    def test_unsatisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b) AND (~a OR ~b)
        ex_unsat1 = ['and',
                    ['or', 'a', 'b'],
                    ['or', ['not', 'a'], 'b'],
                    ['or', 'a', ['not', 'b']],
                    ['or', ['not', 'a'], ['not', 'b']]]
        print(dpll(ex_unsat1))
        
    def test_satisfiable2(self):
        # (a OR b) AND (~a OR c) AND (b OR ~c)
        ex_sat2 = ['and',
                ['or', 'a', 'b'],
                ['or', ['not', 'a'], 'c'],
                ['or', 'b', ['not', 'c']]]
        print(dpll(ex_sat2))
        
    def test_unsatisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i) AND (a OR ~i)
        ex_unsat2 = ['and',
                    ['or', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                    ['or', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'],
                    ['or', ['not', 'a'], ['not', 'i']],
                    ['or', ['not', 'b'], ['not', 'j']],
                    ['or', ['not', 'c'], ['not', 'k']],
                    ['or', ['not', 'd'], ['not', 'l']],
                    ['or', ['not', 'e'], ['not', 'm']],
                    ['or', ['not', 'f'], ['not', 'n']],
                    ['or', ['not', 'g'], ['not', 'o']],
                    ['or', ['not', 'h'], ['not', 'p']],
                    ['or', 'a', 'i'],
                    ['or', ['not', 'a'], 'i'],
                    ['or', 'a', ['not', 'i']]]
        print(dpll(ex_unsat2))

        
    def test_satisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i)
        ex_sat3 = ['and',
                    ['or', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                    ['or', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'],
                    ['or', ['not', 'a'], ['not', 'i']],
                    ['or', ['not', 'b'], ['not', 'j']],
                    ['or', ['not', 'c'], ['not', 'k']],
                    ['or', ['not', 'd'], ['not', 'l']],
                    ['or', ['not', 'e'], ['not', 'm']],
                    ['or', ['not', 'f'], ['not', 'n']],
                    ['or', ['not', 'g'], ['not', 'o']],
                    ['or', ['not', 'h'], ['not', 'p']],
                    ['or', 'a', 'i'],
                    ['or', ['not', 'a'], 'i']]
        print(dpll(ex_sat3))
    
if __name__ == '__main__':
    unittest.main()
    
# results: ???????????????
"""
['a', 'b']
.['b', ['not', 'a']]
.[['not', 'a'], 'i', ['not', 'j'], 'b', ['not', 'c'], ['not', 'd'], ['not', 'e'], ['not', 'f'], ['not', 'g'], ['not', 'h']]
.False
.False
"""