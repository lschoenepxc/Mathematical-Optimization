import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from DifferentiableFunction import DifferentiableFunction
from Function import Function
from DownhillSimplex import DownhillSimplex
from SetsFromFunctions import BoundedSet


class tests_DHSimplex(unittest.TestCase):

    def test_simplex(self):
        R = AffineSpace(1)
        function = Function(name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]))
        startingpoints = np.array([[10.0], [3.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        bounded_set = None
        result = simplex.minimize(function=function, startingpoints=startingpoints, params=params, iterations=100, tol_x=1e-5, tol_y=1e-5, bounded_set=bounded_set)
        
        # Überprüfen, ob das Ergebnis nahe dem globalen Minimum 0 liegt
        expected_result = np.array(0.0)
        np.testing.assert_array_almost_equal(result[0], expected_result, decimal=5)
        
        
    def test_simplex_2d(self):
        interval = MultidimensionalInterval(np.array([-20, -20]), np.array([20,20]))
        function = Function(name="x->(x[0]**2+x[1]**2)", domain=interval, evaluate=lambda x: np.array([x[0]**2+x[1]**2]))
        startingpoints = np.array([[10.0, 0.0], [0.0, 1.0], [10.0, 5.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        bounded_set = None
        result = simplex.minimize(function=function, startingpoints=startingpoints, params=params, iterations=100, tol_x=1e-5, tol_y=1e-5, bounded_set=bounded_set)
        
        # Überprüfen, ob das Ergebnis nahe dem globalen Minimum 0 liegt
        expected_result = np.array(0.0)
        np.testing.assert_array_almost_equal(result[0], expected_result, decimal=2)
        
    def test_simplex_bounded(self):
        interval = MultidimensionalInterval(np.array([-20, -20]), np.array([20,20]))
        function = Function(name="x->(x[0]**2+x[1]**2)", domain=interval, evaluate=lambda x: np.array([x[0]**2+x[1]**2]))
        startingpoints = np.array([[10.0, 4.0], [3.0, 1.0], [10.0, 5.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        inequality_constraints = DifferentiableFunction(name="x->(x[0]-1)", domain=interval, evaluate=lambda x: np.array([x[0]-1]), jacobian=lambda x: np.array([1, 0]))
        bounded_set = BoundedSet(lower_bounds=np.array([1, 1]), upper_bounds=np.array([10, 10]), InequalityConstraints=inequality_constraints)
        result = simplex.minimize(function=function, startingpoints=startingpoints, params=params, iterations=100, tol_x=1e-5, tol_y=1e-5, bounded_set=bounded_set)
        
        # Überprüfen, ob das Ergebnis nahe dem globalen Minimum 1 liegt
        expected_result = np.array(1.0)
        np.testing.assert_array_almost_equal(result[0], expected_result, decimal=2)
        


if __name__ == '__main__':
    unittest.main()
