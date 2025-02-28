import unittest
from Literal import Literal, Clause, CNF
from DPLL import DPLL


class tests_DPLL(unittest.TestCase):

    def test_satisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b)
        a = Literal("a")
        b = Literal("b")
        not_a = -a
        not_b = -b

        clauses = {
            Clause({a, b}),
            Clause({not_a, b}),
            Clause({a, not_b})
        }

        cnf = CNF(clauses)
        dpll = DPLL()
        satisfiable, assignment = dpll.dpll(cnf, {})
        self.assertTrue(satisfiable)
        self.assertEqual(assignment, {'a': True, 'b': True})
        
    def test_unsatisfiable1(self):
        # (a OR b) AND (~a OR b) AND (a OR ~b) AND (~a OR ~b)
        a = Literal("a")
        b = Literal("b")
        not_a = -a
        not_b = -b

        clauses = {
            Clause({a, b}),
            Clause({not_a, b}),
            Clause({a, not_b}),
            Clause({not_a, not_b})
        }

        cnf = CNF(clauses)
        dpll = DPLL()
        satisfiable, assignment = dpll.dpll(cnf, {})
        self.assertFalse(satisfiable)
        self.assertEqual(assignment, {})
        
    def test_satisfiable2(self):
        # (a OR b) AND (~a OR c) AND (b OR ~c)
        a = Literal("a")
        b = Literal("b")
        c = Literal("c")
        not_a = -a
        not_b = -b
        not_c = -c

        clauses = {
            Clause({a, b}),
            Clause({not_a, c}),
            Clause({b, not_c})
        }

        cnf = CNF(clauses)
        dpll = DPLL()
        satisfiable, assignment = dpll.dpll(cnf, {})
        self.assertTrue(satisfiable)
        self.assertIn(assignment, [{'a': True, 'b': True, 'c': True}, {'a': False, 'b': True, 'c': True}, {'a': False, 'b': True, 'c': False}])
    
    def test_unsatisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i) AND (a OR ~i)
        a = Literal("a")
        b = Literal("b")
        c = Literal("c")
        d = Literal("d")
        e = Literal("e")
        f = Literal("f")
        g = Literal("g")
        h = Literal("h")
        i = Literal("i")
        j = Literal("j")
        k = Literal("k")
        l = Literal("l")
        m = Literal("m")
        n = Literal("n")
        o = Literal("o")
        p = Literal("p")
        not_a = -a
        not_b = -b
        not_c = -c
        not_d = -d
        not_e = -e
        not_f = -f
        not_g = -g
        not_h = -h
        not_i = -i
        not_j = -j
        not_k = -k
        not_l = -l
        not_m = -m
        not_n = -n
        not_o = -o
        not_p = -p
        
        clauses = {
            Clause({a, b, c, d, e, f, g, h}),
            Clause({i, j, k, l, m, n, o, p}),
            Clause({not_a, not_i}),
            Clause({not_b, not_j}),
            Clause({not_c, not_k}),
            Clause({not_d, not_l}),
            Clause({not_e, not_m}),
            Clause({not_f, not_n}),
            Clause({not_g, not_o}),
            Clause({not_h, not_p}),
            Clause({a,i}),
            Clause({not_a, i}),
            Clause({a,not_i})
        }
        
        cnf = CNF(clauses)
        dpll = DPLL()
        satisfiable, assignment = dpll.dpll(cnf, {})
        self.assertFalse(satisfiable)
        self.assertEqual(assignment, {})
        
    def test_satisfiable_long(self):
        # (a OR b OR c OR d OR e OR f OR g OR h) AND 
        # (i OR j OR k OR l OR m OR n OR o OR p) AND 
        # (~a OR ~i) AND (~b OR ~j) AND (~c OR ~k) AND 
        # (~d OR ~l) AND (~e OR ~m) AND (~f OR ~n) AND 
        # (~g OR ~o) AND (~h OR ~p) AND 
        # (a OR i) AND (~a OR i)
        a = Literal("a")
        b = Literal("b")
        c = Literal("c")
        d = Literal("d")
        e = Literal("e")
        f = Literal("f")
        g = Literal("g")
        h = Literal("h")
        i = Literal("i")
        j = Literal("j")
        k = Literal("k")
        l = Literal("l")
        m = Literal("m")
        n = Literal("n")
        o = Literal("o")
        p = Literal("p")
        not_a = -a
        not_b = -b
        not_c = -c
        not_d = -d
        not_e = -e
        not_f = -f
        not_g = -g
        not_h = -h
        not_i = -i
        not_j = -j
        not_k = -k
        not_l = -l
        not_m = -m
        not_n = -n
        not_o = -o
        not_p = -p
        
        clauses = {
            Clause({a, b, c, d, e, f, g, h}),
            Clause({i, j, k, l, m, n, o, p}),
            Clause({not_a, not_i}),
            Clause({not_b, not_j}),
            Clause({not_c, not_k}),
            Clause({not_d, not_l}),
            Clause({not_e, not_m}),
            Clause({not_f, not_n}),
            Clause({not_g, not_o}),
            Clause({not_h, not_p}),
            Clause({a,i}),
            Clause({not_a, i})
        }
        
        cnf = CNF(clauses)
        dpll = DPLL()
        satisfiable, assignment = dpll.dpll(cnf, {})
        print(assignment)
        self.assertTrue(satisfiable)
        
        
if __name__ == '__main__':
    unittest.main()
