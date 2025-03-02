import unittest
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from DifferentiableFunction import DifferentiableFunction
from ScaledDifferentiableFunction import ScaledDifferentiableFunction

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
        
        sdf = ScaledDifferentiableFunction()
        
        # Funktion erstellen, die sowohl den Input als auch den Output skaliert und verschiebt
        scaled_offset_f = sdf.getScaledFunction(f, input_scalar=input_scalar, input_offset=input_offset, output_scalar=output_scalar, output_offset=output_offset)
        
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
        
        sdf = ScaledDifferentiableFunction()
        
        # Funktion erstellen, die sowohl den Input als auch den Output skaliert und verschiebt
        scaled_offset_f = sdf.getScaledFunction(f, input_scalar=input_scalar, input_offset=input_offset, output_scalar=output_scalar, output_offset=output_offset)
        
        # Testen, ob die Funktion korrekt skaliert und verschoben wurde
        actual_output = scaled_offset_f.evaluate(1.0)
        expected_output = output_scalar*f.evaluate(input_scalar*1+input_offset)+output_offset
        np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=2)
        
    def test_auto_scaling(self):
        domain = MultidimensionalInterval(np.array([0]), np.array([2]))
        f = DifferentiableFunction(
            name="f",
            domain=domain,
            evaluate=lambda x: x**2,
            jacobian=lambda x: 2*x
        )
        
        sdf = ScaledDifferentiableFunction()
        autoScaled_f = sdf.getAutoScaledFunction(f, BO_option=True)
        
        # Testen, ob die Funktion korrekt skaliert und verschoben wurde
        np.testing.assert_array_almost_equal(autoScaled_f.evaluate(np.array([1.0])), np.array([1.0]), decimal=2)
        np.testing.assert_array_almost_equal(autoScaled_f.evaluate(np.array([-1.0])), np.array([-1.0]), decimal=2)
        np.testing.assert_array_almost_equal(autoScaled_f.evaluate(np.array([0.0])), np.array([-0.5]), decimal=2)
        np.testing.assert_array_almost_equal(autoScaled_f.jacobian(np.array([-1.0])), np.array([0.0]), decimal=2)
        np.testing.assert_array_almost_equal(autoScaled_f.jacobian(np.array([0.0])), np.array([1.0]), decimal=2)
        np.testing.assert_array_almost_equal(autoScaled_f.jacobian(np.array([1.0])), np.array([2.0]), decimal=2)
        self.assertTrue(autoScaled_f._domain._lower_bounds[0] == -1)
        self.assertTrue(autoScaled_f._domain._upper_bounds[0] == 1)
        
        
if __name__ == '__main__':
    unittest.main()