import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction

class tests_ScaledFunction(unittest.TestCase):
    def test_scaling(self):
        R2 = AffineSpace(2)
        f = DifferentiableFunction(
            name="f",
            domain=R2,
            evaluate=lambda x: np.array([x[0]**2, x[1]**2]),
            jacobian=lambda x: np.array([[2*x[0], 0], [0, 2*x[1]]])
        )
        
        # Skalierungsfaktoren und Offsets
        # input_scalar = np.array([2.0])
        # input_offset = np.array([1.0, 1.0])
        # output_scalar = np.array([3.0])
        # output_offset = np.array([1.0, 1.0])
        
        # input_scalar = 2.0
        # input_offset = np.array([1.0, 1.0])
        # output_scalar = np.array([3.0])
        # output_offset = np.array([1.0, 1.0])
        
        input_scalar = 2.0
        input_offset = 0
        output_scalar = np.array([3.0])
        output_offset = np.array([1.0, 1.0])
        
        # Funktion erstellen, die sowohl den Input als auch den Output skaliert und verschiebt
        scaled_offset_f = DifferentiableFunction.getScaledFunction(f, input_scalar=input_scalar, input_offset=input_offset, output_scalar=output_scalar, output_offset=output_offset)
        
        # Testen, ob die Funktion korrekt skaliert und verschoben wurde
        actual_output = scaled_offset_f.evaluate(np.array([1.0, 1.0]))
        expected_output = output_scalar*f.evaluate(input_scalar*np.array([1.0, 1.0])+input_offset)+output_offset
        
        actual_output_jacobian = scaled_offset_f.jacobian(np.array([1.0, 1.0]))
        expected_output_jacobian = output_scalar * f.jacobian(input_scalar * np.array([1.0, 1.0]) + input_offset) * input_scalar
        
        np.testing.assert_array_equal(actual_output, expected_output)
        np.testing.assert_array_equal(actual_output_jacobian, expected_output_jacobian)
        
    def test_scaling2(self):
        R = AffineSpace(1)
        f = DifferentiableFunction(
            name="f",
            domain=R,
            evaluate=lambda x: x**2,
            jacobian=lambda x: 2*x
        )
        
        input_scalar = 2.0
        input_offset = 0
        output_scalar = np.array([3.0])
        output_offset = 0
        
        # Funktion erstellen, die sowohl den Input als auch den Output skaliert und verschiebt
        scaled_offset_f = DifferentiableFunction.getScaledFunction(f, input_scalar=input_scalar, input_offset=input_offset, output_scalar=output_scalar, output_offset=output_offset)
        
        # Testen, ob die Funktion korrekt skaliert und verschoben wurde
        actual_output = scaled_offset_f.evaluate(1.0)
        expected_output = output_scalar*f.evaluate(input_scalar*1+input_offset)+output_offset
        np.testing.assert_array_equal(actual_output, expected_output)
        
        
if __name__ == '__main__':
    unittest.main()