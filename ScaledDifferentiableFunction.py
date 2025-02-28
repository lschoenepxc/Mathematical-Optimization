import numpy as np    
import math
from typing import Union, Callable, Optional
from Set import MultidimensionalInterval
from DifferentiableFunction import DifferentiableFunction
from BayesianOptimization import BO

class ScaledDifferentiableFunction(object):
    def __init__(self):
        super().__init__()
   
    @classmethod
    def getScalingParamsDim(cls, input_scalar: Union[int, float, np.array], output_scalar: Union[int, float, np.array], input_offset: Union[int, float, np.array], output_offset: Union[int, float, np.array], input_dim: int, output_dim: int) -> tuple:
        # Convert scalars to arrays if necessary
        if isinstance(input_scalar, (int, float)):
            input_scalar = np.array([input_scalar])
        if isinstance(input_offset, (int, float)):
            input_offset = np.array([input_offset])
        if isinstance(output_scalar, (int, float)):
            output_scalar = np.array([output_scalar])
        if isinstance(output_offset, (int, float)):
            output_offset = np.array([output_offset])
        # print shapes
        # print("Shapes: ", input_scalar.shape, input_offset.shape, output_scalar.shape, output_offset.shape)
        # Adjust shapes if necessary
        if input_scalar.shape == (1,):
            input_scalar = np.repeat(input_scalar, input_dim)
        if output_scalar.shape == (1,):
            output_scalar = np.repeat(output_scalar, output_dim)
        if input_offset.shape == (1,):
            input_offset = np.repeat(input_offset, input_dim)
        if output_offset.shape == (1,):
            output_offset = np.repeat(output_offset, output_dim)
            
        # print("Shapes: ", input_scalar.shape, input_offset.shape, output_scalar.shape, output_offset.shape)

        # Convert to diagonal matrices
        input_scalar_matrix = np.diag(input_scalar)
        output_scalar_matrix = np.diag(output_scalar)
        # print("Shapes: ", input_scalar_matrix.shape, input_offset.shape, output_scalar_matrix.shape, output_offset.shape)
                    
        assert input_scalar_matrix.shape == (input_dim, input_dim)
        assert input_offset.shape == (input_dim,)
        assert output_scalar_matrix.shape == (output_dim, output_dim)
        assert output_offset.shape == (output_dim,)
        
        return input_scalar_matrix, output_scalar_matrix, input_offset, output_offset, input_scalar, output_scalar
    
    @classmethod
    def getScaledFunction(cls, f: DifferentiableFunction, input_scalar: Optional[Union[int, float, np.array]] = None, output_scalar: Optional[Union[int, float, np.array]] = None, input_offset: Optional[Union[int, float, np.array]] = None, output_offset: Optional[Union[int, float, np.array]] = None) -> DifferentiableFunction:
        """
        Returns a function that scales both input and output and can add offsets to input and output:
        x -> output_scalar * f(input_scalar * x + input_offset) + output_offset
        """
        output_dim = f.evaluate(f.domain.point()).shape[0]
        input_dim = f.domain._ambient_dimension
        
        if input_scalar is None:
            input_scalar = 1
        if output_scalar is None:
            output_scalar = 1
        if input_offset is None:
            input_offset = 0
        if output_offset is None:
            output_offset = 0
        
        input_scalar_matrix, output_scalar_matrix, input_offset, output_offset, input_scalar, output_scalar = cls.getScalingParamsDim(input_scalar, output_scalar, input_offset, output_offset, input_dim, output_dim)
        
        return DifferentiableFunction(
            name=f"{output_scalar} * {f.name}({input_scalar} * x + {input_offset}) + {output_offset}",
            domain=f.domain,
            evaluate=lambda x: np.matmul(output_scalar_matrix, f.evaluate(np.matmul(input_scalar_matrix, x) + input_offset)) + output_offset if isinstance(x, np.ndarray) else output_scalar * f.evaluate(input_scalar * x + input_offset) + output_offset,
            jacobian=lambda x: np.matmul(output_scalar_matrix, np.matmul(f.jacobian(np.matmul(input_scalar_matrix, x) + input_offset), input_scalar_matrix)) if isinstance(x, np.ndarray) else output_scalar * f.jacobian(input_scalar * x + input_offset) * input_scalar
        )
    
    @classmethod
    def getAutoScaledFunction(cls, f: DifferentiableFunction, samples: Optional[int] = 1000, BO_option: Optional[bool] = False):
        """
        Returns the autoscaled function of the given function f, so that the output values are in [0, 1].
        Only implemented for MultidimensionalInterval domains.
        """
        # assert domain is MultiDimensionalInterval
        domain = f.domain
        assert isinstance(domain, MultidimensionalInterval), "The domain of the function must be a MultidimensionalInterval."
        # assert single value output for possible min max calculation
        assert (f.evaluate(f.domain.point())).shape == (1,), "The function should give a single value output"
        
        # BO option
        BO_option = BO_option
        if BO_option:
            # get Min and Max of the function via Bayesian Optimization
            bo = BO()
            # single value output
            min_point = round(float(f.evaluate(bo.Minimize(f))[0]),6)
            max_point = round(float(f.evaluate(bo.Minimize((-1)*f))[0]),6)
        else:
            # get Min and Max of the function via Sampling
            samples = samples
            x = np.array([domain.point() for i in range(samples)])
            y = np.array([f.evaluate(x[i]) for i in range(samples)])
            min_point = round(float(np.min(y)),6)
            max_point = round(float(np.max(y)),6)
            
        # print("Min: ", min_point, "Max: ", max_point)
        
        # calculate the output scalar and offset so that the output is scaled to [0, 1]
        output_scalar = 1 / (max_point - min_point)
        output_offset = -min_point / (max_point - min_point)
        
        # print("Params: ", output_scalar, output_offset)
        
        return cls.getScaledFunction(f, output_scalar=output_scalar, output_offset=output_offset)