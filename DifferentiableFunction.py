from abc import ABC, abstractmethod
from Set import ISet, AffineSpace
import numpy as np
from typing import Callable, Union
from multimethod import multimethod
from Function import IFunction, Function


class IDifferentiableFunction(IFunction):
    """This interface models differentiable Functions from ISet to R^n."""

    def __init__(self, name: str, domain: ISet):
        # super().__init__()
        # self._name = name
        # self._domain = domain
        super().__init__(name=name, domain=domain)

    # @abstractmethod
    # def evaluate(self, point: np.ndarray) -> np.ndarray:
    #     """Evaluates the function at point.
    #     The parameter "point" is a vector in Double^(Domain.ambient_dimension) such that Domain.is_contained(point)=true."""
    #     pass

    # @property
    # def domain(self) -> ISet:
    #     """The domain of the function, i.e., the set of points at which the function can be evaluated."""
    #     return self._domain

    # @property
    # def name(self) -> str:
    #     """The name of the function, might be used for debugging"""
    #     return self._name

    @abstractmethod
    def jacobian(self, point: np.ndarray) -> np.ndarray:
        """This is the evaluated Jacobian of this function f at a point.
        The parameter "point" is a vector in Double^(Domain.ambient_dimension) such that Domain.is_contained(point)=true."""
        pass

    @multimethod
    def __add__(self, other: Union[int, float]) -> 'IDifferentiableFunction':
        """Adds the two functions value wise, where the second function is a constant"""
        added_function = Function.__add__(self, other)
        return DifferentiableFunction(
            name=added_function.name,
            domain=added_function.domain,
            evaluate=added_function.evaluate,
            jacobian=lambda v: self.jacobian(v)
        )
        # return DifferentiableFunction(
        #     name="(" + self.name + ") + " + str(other),
        #     domain=self.domain,
        #     evaluate=lambda v: self.evaluate(v) + other,
        #     jacobian=lambda v: self.jacobian(v)
        # )

    @multimethod
    def __add__(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Adds the two functions value wise"""
        added_function = Function.__add__(self, other)
        return DifferentiableFunction(
            name=added_function.name,
            domain=added_function.domain,
            evaluate=added_function.evaluate,
            jacobian=lambda v: self.jacobian(v) + other.jacobian(v)
        )
        # return DifferentiableFunction(
        #     name="(" + self.name + ") + (" + other.name + ")",
        #     domain=self.domain.intersect(other.domain),
        #     evaluate=lambda v: self.evaluate(v) + other.evaluate(v),
        #     jacobian=lambda v: self.jacobian(v) + other.jacobian(v)
        # )

    def __mul__(self, other: Union[int, float]) -> 'IDifferentiableFunction':
        """Multiplies the function by a scalar"""
        multiplied_function = Function.__mul__(self, other)
        return DifferentiableFunction(
            name=multiplied_function.name,
            domain=multiplied_function.domain,
            evaluate=multiplied_function.evaluate,
            jacobian=lambda v: other * self.jacobian(v)
        )
        # return DifferentiableFunction(
        #     name=str(other) + " * (" + self.name + ")",
        #     domain=self.domain,
        #     evaluate=lambda v: other * self.evaluate(v),
        #     jacobian=lambda v: other * self.jacobian(v)
        # )

    def __pow__(self, power: int) -> 'IDifferentiableFunction':
        """Take integer exponents of a function"""
        powered_function = Function.__pow__(self, power)
        return DifferentiableFunction(
            name=powered_function.name,
            domain=powered_function.domain,
            evaluate=powered_function.evaluate,
            jacobian=lambda v: np.matmul(
                np.array([[power]])*self.evaluate(v)**(power-1), self.jacobian(v))
        )
        # return DifferentiableFunction(
        #     name="(" + self.name + ")^" + str(power),
        #     domain=self.domain,
        #     evaluate=lambda v: self.evaluate(v)**power,
        #     jacobian=lambda v: np.matmul(
        #         np.array([[power]])*self.evaluate(v)**(power-1), self.jacobian(v))
        # )

    def __rmul__(self, other: Union[int, float]):
        """Multiplies the function by a scalar"""
        return self.__mul__(other)

    # warum funktioniert das ohne eine neue Funktion zu erstellen?
    def __sub__(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Subtracts two functions value wise"""
        return self + (-1) * other

    def Pairing(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Returns the pairing of two functions"""
        paired_function = Function.Pairing(self, other)
        return DifferentiableFunction(
            name=paired_function.name,
            domain=paired_function.domain,
            evaluate=paired_function.evaluate,
            jacobian=lambda v: np.concatenate(
                (self.jacobian(v), other.jacobian(v)))
        )
        # return DifferentiableFunction(
        #     name="Pair(" + self.name + "," + other.name + ")",
        #     domain=self.domain.intersect(other.domain),
        #     evaluate=lambda v: np.concatenate(
        #         (self.evaluate(v), other.evaluate(v))),
        #     jacobian=lambda v: np.concatenate(
        #         (self.jacobian(v), other.jacobian(v)))
        # )

    def CartesianProduct(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        """Returns the cartesian product of two functions"""

        composed_function = Function.CartesianProduct(self, other)
        proj_domain = AffineSpace(
            self.domain._ambient_dimension+other.domain._ambient_dimension)
        proj1 = DifferentiableFunction.Projection(
            domain=proj_domain, l=range(0, self.domain._ambient_dimension))
        proj2 = DifferentiableFunction.Projection(
            domain=proj_domain, l=range(self.domain._ambient_dimension, self.domain._ambient_dimension+other.domain._ambient_dimension))
        f1 = DifferentiableFunction.FromComposition(
            self, proj1)
        f2 = DifferentiableFunction.FromComposition(
            other, proj2)
        return DifferentiableFunction(
            name=composed_function.name,
            domain=composed_function.domain,
            evaluate=composed_function.evaluate,
            jacobian=lambda v: np.concatenate(
                (f1.jacobian(v), f2.jacobian(v)))
        )
        # return DifferentiableFunction(
        #     name="CartesianProduct(" + self.name + "," + other.name + ")",
        #     domain=self.domain.cartesian_product(other.domain),
        #     evaluate=lambda v: np.concatenate(
        #         (f1.evaluate(v), f2.evaluate(v))),
        #     jacobian=lambda v: np.concatenate(
        #         (f1.jacobian(v), f2.jacobian(v)), axis=0)
        # )


class DifferentiableFunction(Function, IDifferentiableFunction):
    """This class models differentiable Functions from ISet to R^n, where the function and the Jacobian are given by lambdas."""

    def __init__(self, name: str, domain: ISet, evaluate: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray]):
        """ Construct a function from lambdas, for the function itself and for its derivatives
        Sadly, the type system is not strict enough to check sizes of tensors"""
        super().__init__(name=name, domain=domain, evaluate=evaluate)
        # self._evaluate = evaluate
        self._jacobian = jacobian

    # def evaluate(self, point: np.ndarray) -> np.ndarray:
    #     return self._evaluate(point)

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        return self._jacobian(point)

    @ classmethod
    def FromComposition(cls, f: IDifferentiableFunction, g: IDifferentiableFunction) -> IDifferentiableFunction:
        """Constructs f ° g"""
        composed_function = Function.FromComposition(f, g)
        return cls(
            name=composed_function.name,
            domain=composed_function.domain,
            evaluate=composed_function.evaluate,
            jacobian=lambda v: np.matmul(
                f.jacobian(g.evaluate(v)), g.jacobian(v))
        )
        # return cls(
        #     name="(" + f.name + ") ° (" + g.name + ")",
        #     domain=g.domain,
        #     evaluate=lambda v: f.evaluate(g.evaluate(v)),
        #     jacobian=lambda v: np.matmul(
        #         f.jacobian(g.evaluate(v)), g.jacobian(v))
        # )

    @ classmethod
    def LinearMapFromMatrix(cls, A: np.array) -> IDifferentiableFunction:
        """Constructs x -> A*x"""
        linear_function = Function.LinearMapFromMatrix(A)
        return cls(
            name=linear_function.name,
            domain=linear_function.domain,
            evaluate=linear_function.evaluate,
            jacobian=lambda x: A
        )
        # return cls(
        #     name="linear",
        #     domain=AffineSpace(A.shape[1]),
        #     evaluate=lambda x: np.matmul(A, x),
        #     jacobian=lambda x: A
        # )

    @classmethod
    def TranslationByVector(cls, v: np.array) -> IDifferentiableFunction:
        """Constructs x -> x+v"""
        n = v.shape[0]
        translated_function = Function.TranslationByVector(v)
        return cls(
            name=translated_function.name,
            domain=translated_function.domain,
            evaluate=translated_function.evaluate,
            jacobian=lambda x: np.eye(n)
        )
        # return cls(name="translation", domain=AffineSpace(n), evaluate=lambda x: x+v, jacobian=lambda x: np.eye(n))

    def __create_matrix_with_ones(rows: int, columns: int, ones_positions: list[tuple[int, int]]):
        matrix = np.zeros((rows, columns))
        row_indices, column_indices = zip(*ones_positions)
        matrix[row_indices, column_indices] = 1
        return matrix

    @ classmethod
    def Projection(cls, domain: ISet, l: list[int]) -> IDifferentiableFunction:
        """Constructs a projection function"""
        projected_function = Function.Projection(domain, l)
        return cls(
            name=projected_function.name,
            domain=projected_function.domain,
            evaluate=projected_function.evaluate,
            jacobian=lambda x: cls.__create_matrix_with_ones(rows=len(l), columns=domain._ambient_dimension, ones_positions=list(zip(range(len(l)), l)))
        )
        # return cls(
        #     name="projection("+str(l)+")",
        #     domain=domain,
        #     evaluate=lambda x: x[l],
        #     jacobian=lambda x: cls.__create_matrix_with_ones(rows=len(l), columns=domain._ambient_dimension, ones_positions=list(zip(range(len(l)), l)))
        # )

    @classmethod
    def Identity(cls, domain: ISet) -> IDifferentiableFunction:
        n = domain._ambient_dimension
        identity_function = Function.Identity(domain)
        return cls(
            name=identity_function.name,
            domain=identity_function.domain,
            evaluate=identity_function.evaluate,
            jacobian=lambda x: np.eye(n)
        )
        # return cls(
        #     name="Id(n)",
        #     domain=domain,
        #     evaluate=lambda x: x,
        #     jacobian=lambda x: np.eye(n))

    @classmethod
    def ReLU(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a ReLU function"""
        return cls(name="ReLU", domain=AffineSpace(dimension), evaluate=lambda x: np.maximum(0, x), jacobian=lambda x: np.diag(x >= 0))

    @classmethod
    def TwoNormSquared(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a function computing the squared two norm of a vector"""
        return cls(name="TwoNormSquared", domain=AffineSpace(dimension), evaluate=lambda x: np.linalg.norm(x)**2, jacobian=lambda x: (2*x).reshape(1, dimension))

    @classmethod
    def Debug(cls, f: IDifferentiableFunction) -> IDifferentiableFunction:
        """Returns a modified function that prints its inputs and outputs"""
        def modified_evaluate(x):
            result = f.evaluate(x)
            print(f"{f.name} - Input: {x}, Output: {result}")
            return result
        return cls(name=f.name, domain=f.domain, evaluate=modified_evaluate, jacobian=f.jacobian)
    
        
    @classmethod
    def getScaledFunction(cls, f: IDifferentiableFunction, input_scalar: Union[int, float, np.array], output_scalar: Union[int, float, np.array], input_offset: Union[int, float, np.array], output_offset: Union[int, float, np.array]) -> IDifferentiableFunction:
        """
        Returns a function that scales both input and output and can add offsets to input and output:
        x -> output_scalar * f(input_scalar * x + input_offset) + output_offset
        """
        output_dim = f.evaluate(np.zeros(f.domain._ambient_dimension)).shape[0]
        input_dim = f.domain._ambient_dimension
        
        input_scalar_matrix, output_scalar_matrix, input_offset, output_offset, input_scalar, output_scalar = cls.getScalingParamsDim(input_scalar, output_scalar, input_offset, output_offset, input_dim, output_dim)
        scaled_function = Function.getScaledFunction(f, input_scalar, output_scalar, input_offset, output_offset)
        return cls(
            name=scaled_function.name,
            domain=scaled_function.domain,
            evaluate=scaled_function.evaluate,
            jacobian=lambda x: np.matmul(output_scalar_matrix, np.matmul(f.jacobian(np.matmul(input_scalar_matrix, x) + input_offset), input_scalar_matrix)) if isinstance(x, np.ndarray) else output_scalar * f.jacobian(input_scalar * x + input_offset) * input_scalar
        )