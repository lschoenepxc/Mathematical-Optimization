import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from DifferentiableFunction import DifferentiableFunction
from Function import Function
from DownhillSimplex import DownhillSimplex


class tests_DHSimplex(unittest.TestCase):

    def test_simplex(self):
        R = AffineSpace(1)
        function = Function(name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]))
        startingpoints = np.array([[10.0], [3.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        result = simplex.minimize(function, startingpoints, params, iterations=100, tol_x=1e-5, tol_y=1e-5)
        
        # Überprüfen, ob das Ergebnis nahe dem globalen Minimum [0, 0] liegt
        expected_result = np.array(0.0)
        np.testing.assert_array_almost_equal(result[0], expected_result, decimal=5)
        
        
    def test_simplex_2d(self):
        interval = MultidimensionalInterval(np.array([-20, -20]), np.array([20,20]))
        function = Function(name="x->(x[0]**2+x[1]**2)", domain=interval, evaluate=lambda x: np.array([x[0]**2+x[1]**2]))
        startingpoints = np.array([[10.0, 0.0], [0.0, 1.0], [10.0, 5.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        #self, function: IFunction, startingpoints: np.array, params: dict={'alpha':1.0, 'gamma':2.0, 'beta': 0.5, 'delta': 0.5}, iterations: int = 100, tol_x=1e-5, tol_y=1e-5
        result = simplex.minimize(function, startingpoints, params, iterations=10000, tol_x=1e-5, tol_y=1e-5)
        
        # Überprüfen, ob das Ergebnis nahe dem globalen Minimum [0, 0] liegt
        expected_result = np.array(0.0)
        np.testing.assert_array_almost_equal(result[0], expected_result, decimal=2)
        


if __name__ == '__main__':
    unittest.main()
