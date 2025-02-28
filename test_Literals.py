import unittest
from Literal import Literal, Clause, CNF


class tests_Literals(unittest.TestCase):

    def test_literal(self):
        a = Literal("a")
        b = Literal("b")
        not_a = -a
        not_b = -b

        self.assertEqual(a.__hash__(), hash(('a', False)))
        self.assertEqual(a.__repr__(), "a")
        self.assertEqual(a.evaluate({'a': True}), True)
        self.assertEqual(a.evaluate({'a': False}), False)
        self.assertEqual(a.evaluate({}), False)

        self.assertEqual(b.__hash__(), hash(('b', False)))
        self.assertEqual(b.__repr__(), "b")
        self.assertEqual(b.evaluate({'b': True}), True)
        self.assertEqual(b.evaluate({'b': False}), False)
        self.assertEqual(b.evaluate({}), False)

        self.assertEqual(not_a.__hash__(), hash(('a', True)))
        self.assertEqual(not_a.__repr__(), "~a")
        self.assertEqual(not_a.evaluate({'a': True}), False)
        self.assertEqual(not_a.evaluate({'a': False}), True)
        self.assertEqual(not_a.evaluate({}), True)

        self.assertEqual(not_b.__hash__(), hash(('b', True)))
        self.assertEqual(not_b.__repr__(), "~b")
        self.assertEqual(not_b.evaluate({'b': True}), False)
        self.assertEqual(not_b.evaluate({'b': False}), True)
        self.assertEqual(not_b.evaluate({}), True)
    
    def test_clause(self):
        a = Literal("a")
        b = Literal("b")
        not_a = -a
        not_b = -b

        clause = Clause({a, b})
        self.assertEqual(clause.__hash__(), hash(frozenset({a, b})))
        self.assertIn(clause.__repr__(), {"a | b", "b | a"})
        self.assertEqual(clause.evaluate({'a': True, 'b': True}), True)
        self.assertEqual(clause.evaluate({'a': True, 'b': False}), True)
        self.assertEqual(clause.evaluate({'a': False, 'b': True}), True)
        self.assertEqual(clause.evaluate({'a': False, 'b': False}), False)
        self.assertEqual(clause.evaluate({}), False)

        clause = Clause({not_a, b})
        self.assertEqual(clause.__hash__(), hash(frozenset({not_a, b})))
        self.assertIn(clause.__repr__(), {"b | ~a", "~a | b"})
        self.assertEqual(clause.evaluate({'a': True, 'b': True}), True)
        self.assertEqual(clause.evaluate({'a': True, 'b': False}), False)
        self.assertEqual(clause.evaluate({'a': False, 'b': True}), True)
        self.assertEqual(clause.evaluate({'a': False, 'b': False}), True)
        self.assertEqual(clause.evaluate({}), True)

        clause = Clause({a, not_b})
        self.assertEqual(clause.__hash__(), hash(frozenset({a, not_b})))
        self.assertIn(clause.__repr__(), {"~b | a", "a | ~b"})
        self.assertEqual(clause.evaluate({'a': True, 'b': True}), True)
        self.assertEqual(clause.evaluate({'a': True, 'b': False}), True)
        self.assertEqual(clause.evaluate({'a': False, 'b': True}), False)
        self.assertEqual(clause.evaluate({'a': False, 'b': False}), True)
        self.assertEqual(clause.evaluate({}), True)
        
    def test_CNF(self):
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
        self.assertEqual(cnf.__hash__(), hash(frozenset(clauses)))
        # not applicable, because the order of the clauses and literals is not defined
        # self.assertIn(cnf.__repr__(), {"a | b & b | ~a & a | ~b", "b | ~a & a | b & a | ~b", "b | ~a & a | ~b & a | b", ...})
        self.assertEqual(cnf.evaluate({'a': True, 'b': True}), True)
        self.assertEqual(cnf.evaluate({'a': True, 'b': False}), False)
        self.assertEqual(cnf.evaluate({'a': False, 'b': True}), False)
        self.assertEqual(cnf.evaluate({'a': False, 'b': False}), False)
        self.assertEqual(cnf.evaluate({}), False)

        clauses = {
            Clause({a, b}),
            Clause({not_a, b}),
            Clause({a, not_b}),
            Clause({not_a, not_b})
        }
        cnf = CNF(clauses)
        self.assertEqual(cnf.__hash__(), hash(frozenset(clauses)))
        # not applicable, because the order of the clauses and literals is not defined
        # self.assertIn(cnf.__repr__(), {"a | b & b | ~a & a | ~b & ~a | ~b", "b | ~a & a | b & a | ~b & ~a | ~b, ..."})
        self.assertEqual(cnf.evaluate({'a': True, 'b': True}), False)
        self.assertEqual(cnf.evaluate({'a': True, 'b': False}), False)
        self.assertEqual(cnf.evaluate({'a': False, 'b': True}), False)
        self.assertEqual(cnf.evaluate({'a': False, 'b': False}), False)
        self.assertEqual(cnf.evaluate({}), False)
        
        
if __name__ == '__main__':
    unittest.main()
