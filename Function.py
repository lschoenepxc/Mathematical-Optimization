from abc import ABC, abstractmethod
from Set import ISet, AffineSpace
import numpy as np
from typing import Callable, Union
from multimethod import multimethod


class IFunction(object):
    """This interface models Functions from ISet to R^n."""

    def __init__(self, name: str, domain: ISet, output_domain: ISet):
        super().__init__()
        self._name = name
        self._domain = domain
        self._output_domain = output_domain
        self._verfiied = False

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
    
    @property
    def output_domain(self) -> ISet:
        """The domain of the function, i.e., the set of points at which the function can be evaluated."""
        return self._output_domain


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
        
    def verfify(self):
        """Verifies that the function is well defined"""
        if not self._verfiied:
            assert self._output_domain.shape == self.evaluate(self.domain.point()).shape , "Function is not well defined"
            self._verfiied = True

    def __mul__(self, other: Union[int, float]) -> 'IFunction':
        """Multiplies the function by a scalar"""
        return Function(
            name=str(other) + " * (" + self.name + ")",
            domain=self.domain,
            evaluate=lambda v: other * self.evaluate(v)
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
    """This class models Functions from ISet to R^n, where the function is given by lambdas."""

    def __init__(self, name: str, domain: ISet, evaluate: Callable[[np.ndarray], np.ndarray]):
        """ Construct a function from lambdas, for the function itself
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

    def __create_matrix_with_ones(rows: int, columns: int, ones_positions: list[tuple[int, int]]):
        matrix = np.zeros((rows, columns))
        row_indices, column_indices = zip(*ones_positions)
        matrix[row_indices, column_indices] = 1
        return matrix

    @ classmethod
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
            evaluate=lambda x: x)

    @classmethod
    def ReLU(cls, dimension: int) -> IFunction:
        """Returns a ReLU function"""
        return cls(name="ReLU", domain=AffineSpace(dimension), evaluate=lambda x: np.maximum(0, x))

    @classmethod
    def TwoNormSquared(cls, dimension: int) -> IFunction:
        """Returns a function computing the squared two norm of a vector"""
        return cls(name="TwoNormSquared", domain=AffineSpace(dimension), evaluate=lambda x: np.linalg.norm(x)**2)

    @classmethod
    def Debug(cls, f: IFunction) -> IFunction:
        """Returns a modified function that prints its inputs and outputs"""
        def modified_evaluate(x):
            result = f.evaluate(x)
            print(f"{f.name} - Input: {x}, Output: {result}")
            return result
        return cls(name=f.name, domain=f.domain, evaluate=modified_evaluate)
