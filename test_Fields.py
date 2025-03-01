from Set import RealNumbers, MultidimensionalInterval
from Field import RealNumberField, VectorSpace, LinearMap, gauss_jordan_algorithm
import numpy as np
import unittest


class tests_fields(unittest.TestCase):
    def test_real_numbers(self):
        real_numbers = RealNumbers()
        self.assertTrue(real_numbers.contains(3.0))
        self.assertFalse(real_numbers.contains("not a number"))
    
    def test_real_numbers_field(self):
        real_numbers = RealNumbers()
        field = RealNumberField(real_numbers)
        
        a = 3.0
        b = 4.0
        
        self.assertEqual(field.add(a, b), 7.0)
        self.assertEqual(field.mul(a, b), 12.0)
        self.assertEqual(field.sub(a, b), -1.0)
        self.assertEqual(field.div(a, b), 0.75)
        self.assertEqual(field.neg(a), -3.0)
        self.assertEqual(field.inv(a), 1/3)
        self.assertEqual(field.zero(), 0.0)
        self.assertEqual(field.one(), 1.0)
        
        with self.assertRaises(ValueError):
            c = "not a number"
            field.add(a, c)
    
    def test_vector_space(self):
        real_numbers = RealNumbers()
        field = RealNumberField(real_numbers)
        vector_space = VectorSpace(field)
        
        v = [1.0, 2.0, 3.0]
        w = [4.0, 5.0, 6.0]
        
        self.assertEqual(vector_space.add(v, w), [5.0, 7.0, 9.0])
        self.assertEqual(vector_space.scalar_mul(2.0, v), [2.0, 4.0, 6.0])
        self.assertEqual(vector_space.zero_vector(len(v)), [0.0, 0.0, 0.0])
        
    def test_linear_map(self):
        real_numbers = RealNumbers()
        field = RealNumberField(real_numbers)
        vector_space = VectorSpace(field)
        
        matrix = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        linear_map = LinearMap(vector_space, vector_space, matrix)
        
        v = [1, 2, 3]
        
        self.assertEqual(linear_map.apply(v), [14.0, 14.0, 17.0])
        self.assertTrue(np.array_equal(linear_map.matrix_representation(), np.array(matrix)))
        
        A = LinearMap(vector_space, vector_space, [[1, 2], [3, 4]])
        v = [5, 6]
        B = [[7, 8], [9, 10]]
        
        self.assertEqual(A.apply(v), [17.0, 39.0])
        self.assertTrue(np.array_equal(A.apply(B), np.dot(A.matrix_representation(), B)))
    
    def test_gauss_inverse(self):
        matrix = [
            [2, 1, 1],
            [1, 3, 2],
            [1, 0, 0]
        ]
        inverse_known = [
            [0, 0, 1],
            [-2, 1, 3],
            [3, -1, -5]
        ]
        
        assert np.array_equal(np.matmul(matrix, inverse_known), np.eye(3))
        
        gauss = gauss_jordan_algorithm(matrix)
        gauss.solve()
        inverse = gauss.compute_inverse()

        # assert that the inverse matrix is correct by 3 decimal places
        assert np.allclose(np.matmul(matrix, inverse), np.eye(3), atol=1e-3)
        
    def test_gauss(self):
        matrix = [
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]
        ]
        b = [8, -11, -3]

        gauss = gauss_jordan_algorithm(matrix, b)
        solution = gauss.solve()
        
        assert np.allclose(np.matmul(matrix, solution), b, atol=1e-3)
        
    
if __name__ == '__main__':
    unittest.main()