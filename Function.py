from abc import ABC, abstractmethod
from Set import ISet, AffineSpace
import numpy as np
from typing import Callable, Union
from multimethod import multimethod

class IFunction(object):
    """This interface models differentiable Functions from ISet to R^n."""
    def __init__(self, name: str, domain: ISet):
        super().__init__()
        self._name = name
        self._domain = domain

    @abstractmethod
    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Evaluates the function at point.
        The parameter "point" is a vector in Double^(Domain.ambient_dimension) such that Domain.is_contained(point)=true."""
        pass

    @property
    def domain(self) -> ISet:
        """The domain of the function, i.e., the set of points at which the function can be evaluated."""
        return self._domain

    @property
    def name(self) -> str:
        """The name of the function, might be used for debugging"""
        return self._name
    
    @multimethod
    def __add__(self, other: Union[int, float]) -> 'IFunction':
        """Adds the two functions value wise, where the second function is a constant"""
        return Function(
            name="(" + self.name + ") + " + str(other),
            domain=self.domain,
            evaluate=lambda v: self.evaluate(v) + other
        )
    
    @multimethod
    def __add__(self, other: 'IFunction') -> 'IFunction':
        """Adds the two functions value wise"""
        return Function(
            name="(" + self.name + ") + (" + other.name + ")",
            domain=self.domain.intersect(other.domain),
            evaluate=lambda v: self.evaluate(v) + other.evaluate(v)
        )
    
    @multimethod
    def __mul__(self, other: Union[int, float]) -> 'IFunction':
        """Multiplies the function by a scalar"""
        return Function(
            name=str(other) + " * (" + self.name + ")",
            domain=self.domain,
            evaluate=lambda v: other * self.evaluate(v)
        )
    
    @multimethod
    def __mul__(self, other: 'IFunction') -> 'IFunction':
        """Multiplies the two functions value wise"""
        assert self.evaluate(np.zeros(self.domain._ambient_dimension)).shape == other.evaluate(np.zeros(other.domain._ambient_dimension)).shape, "The two functions must have the same output dimension"
        return Function(
            name="(" + self.name + ") * (" + other.name + ")",
            domain=self.domain.intersect(other.domain),
            evaluate=lambda v: np.multiply(self.evaluate(v), other.evaluate(v))
        )

    def __pow__(self, power: int) -> 'IFunction':
        """Take integer exponents of a function"""
        return Function(
            name="(" + self.name + ")^" + str(power),
            domain=self.domain,
            evaluate=lambda v: self.evaluate(v)**power
        )

    def __rmul__(self, other: Union[int, float]):
        """Multiplies the function by a scalar"""
        return self.__mul__(other)

    def __sub__(self, other: 'IFunction') -> 'IFunction':
        """Subtracts two functions value wise"""
        return self + (-1) * other

    def Pairing(self, other: 'IFunction') -> 'IFunction':
        """Returns the pairing of two functions"""
        return Function(
            name="Pair(" + self.name + "," + other.name + ")",
            domain=self.domain.intersect(other.domain),
            evaluate=lambda v: np.concatenate(
                (self.evaluate(v), other.evaluate(v)))
        )

    def CartesianProduct(self, other: 'IFunction') -> 'IFunction':
        """Returns the cartesian product of two functions"""

        proj_domain = AffineSpace(
            self.domain._ambient_dimension+other.domain._ambient_dimension)
        proj1 = Function.Projection(
            domain=proj_domain, l=range(0, self.domain._ambient_dimension))
        proj2 = Function.Projection(
            domain=proj_domain, l=range(self.domain._ambient_dimension, self.domain._ambient_dimension+other.domain._ambient_dimension))
        f1 = Function.FromComposition(
            self, proj1)
        f2 = Function.FromComposition(
            other, proj2)
        return Function(
            name="CartesianProduct(" + self.name + "," + other.name + ")",
            domain=self.domain.cartesian_product(other.domain),
            evaluate=lambda v: np.concatenate(
                (f1.evaluate(v), f2.evaluate(v)))
        )
    
class Function(IFunction):
    """This class models differentiable Functions from ISet to R^n, where the function and the Jacobian are given by lambdas."""

    def __init__(self, name: str, domain: ISet, evaluate: Callable[[np.ndarray], np.ndarray]):
        """ Construct a function from lambdas, for the function itself and for its derivatives
        Sadly, the type system is not strict enough to check sizes of tensors"""
        super().__init__(name=name, domain=domain)
        self._evaluate = evaluate

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        return self._evaluate(point)
    
    @ classmethod
    def FromComposition(cls, f: IFunction, g: IFunction) -> IFunction:
        """Constructs f ° g"""
        return cls(
            name="(" + f.name + ") ° (" + g.name + ")",
            domain=g.domain,
            evaluate=lambda v: f.evaluate(g.evaluate(v))
        )
        
    @ classmethod
    def LinearMapFromMatrix(cls, A: np.array) -> IFunction:
        """Constructs x -> A*x"""
        return cls(
            name="linear",
            domain=AffineSpace(A.shape[1]),
            evaluate=lambda x: np.matmul(A, x)
        )
        
    @classmethod
    def TranslationByVector(cls, v: np.array) -> IFunction:
        """Constructs x -> x+v"""
        n = v.shape[0]
        return cls(name="translation", domain=AffineSpace(n), evaluate=lambda x: x+v)
    
    @classmethod
    def Projection(cls, domain: ISet, l: list[int]) -> IFunction:
        return cls(
            name="projection("+str(l)+")",
            domain=domain,
            evaluate=lambda x: x[l]
        )
    
    @classmethod
    def Identity(cls, domain: ISet) -> IFunction:
        n = domain._ambient_dimension
        return cls(
            name="Id(n)",
            domain=domain,
            evaluate=lambda x: x
        )
    
    @classmethod
    def Debug(cls, f: IFunction) -> IFunction:
        """Returns a modified function that prints its inputs and outputs"""
        def modified_evaluate(x):
            result = f.evaluate(x)
            print(f"{f.name} - Input: {x}, Output: {result}")
            return result
        return cls(name=f.name, domain=f.domain, evaluate=modified_evaluate)
    
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
    def getScaledFunction(cls, f: IFunction, input_scalar: Union[int, float, np.array], output_scalar: Union[int, float, np.array], input_offset: Union[int, float, np.array], output_offset: Union[int, float, np.array]) -> IFunction:
        """
        Returns a function that scales both input and output and can add offsets to input and output:
        x -> output_scalar * f(input_scalar * x + input_offset) + output_offset
        """
        output_dim = f.evaluate(np.zeros(f.domain._ambient_dimension)).shape[0]
        input_dim = f.domain._ambient_dimension
        
        input_scalar_matrix, output_scalar_matrix, input_offset, output_offset, input_scalar, output_scalar = cls.getScalingParamsDim(input_scalar, output_scalar, input_offset, output_offset, input_dim, output_dim)

        return cls(
            name=f"{output_scalar} * {f.name}({input_scalar} * x + {input_offset}) + {output_offset}",
            domain=f.domain,
            evaluate=lambda x: np.matmul(output_scalar_matrix, f.evaluate(np.matmul(input_scalar_matrix, x) + input_offset)) + output_offset if isinstance(x, np.ndarray) else output_scalar * f.evaluate(input_scalar * x + input_offset) + output_offset
        )
        
    ### Aufgabe 4.2: Implemetiere 10 weitere Funktionen: sin, cos, tan, exp, log, sqrt, sigmoid, heaviside, square, cube
    @classmethod
    def sin(cls, dimension: int) -> IFunction:
        """Returns a sinus function"""
        return cls(name="sinus", domain=AffineSpace(dimension), evaluate=lambda x: np.sin(x))
    
    @classmethod
    def cos(cls, dimension: int) -> IFunction:
        """Returns a cosinus function"""
        return cls(name="cosinus", domain=AffineSpace(dimension), evaluate=lambda x: np.cos(x))
    
    @classmethod
    def tan(cls, dimension: int) -> IFunction:
        """Returns a tan function"""
        return cls(name="tan", domain=AffineSpace(dimension), evaluate=lambda x: np.tan(x))
    
    @classmethod
    def exp(cls, dimension: int) -> IFunction:
        """Returns a exponential function"""
        return cls(name="exp", domain=AffineSpace(dimension), evaluate=lambda x: np.exp(x))
    
    @classmethod
    def log(cls, dimension: int) -> IFunction:
        """Returns a logarithm function"""
        return cls(name="log", domain=AffineSpace(dimension), evaluate=lambda x: np.log(x))
    
    @classmethod
    def sqrt(cls, dimension: int) -> IFunction:
        """Returns a square root function"""
        return cls(name="sqrt", domain=AffineSpace(dimension), evaluate=lambda x: np.sqrt(x))
    
    @classmethod
    def sigmoid(cls, dimension: int) -> IFunction:
        """Returns a sigmoid function"""
        return cls(name="sigmoid", domain=AffineSpace(dimension), evaluate=lambda x: 1/(1+np.exp(-x)))
    
    @classmethod
    def square(cls, dimension: int) -> IFunction:
        """Returns a square function"""
        return cls(name="square", domain=AffineSpace(dimension), evaluate=lambda x: x**2)
    
    @classmethod
    def cube(cls, dimension: int) -> IFunction:
        """Returns a cube function"""
        return cls(name="cube", domain=AffineSpace(dimension), evaluate=lambda x: x**3)
    
    @classmethod
    def arccos(cls, dimension: int) -> IFunction:
        """Returns a arccos function"""
        return cls(name="arccos", domain=AffineSpace(dimension), evaluate=lambda x: np.arccos(x))
    
    ### Ende Aufgabe 4.2