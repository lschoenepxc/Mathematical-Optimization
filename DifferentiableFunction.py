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

    @multimethod
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
    
    @multimethod
    def __mul__(self, other: 'IDifferentiableFunction') -> 'IDifferentiableFunction':
        # only for componentwise evaluating, one dimensional functions
        
        multiplied_function = Function.__mul__(self, other)
        
        # Überprüfen, ob eine Jacobian-Matrix die Shape (1,) und die andere die Shape (1,1) hat
        if (self.jacobian(self.domain.point()).shape == (1,) and other.jacobian(other.domain.point()).shape == (1, 1)) or \
        (self.jacobian(self.domain.point()).shape == (1, 1) and other.jacobian(other.domain.point()).shape == (1,)) or \
        (self.jacobian(self.domain.point()).shape == (1,) and other.jacobian(other.domain.point()).shape == (1,)):
            jacobian=lambda v: np.matmul(self.jacobian(v).reshape(1, 1), other.evaluate(v)) + np.matmul(self.evaluate(v), other.jacobian(v).reshape(1, 1))

        else :
            jacobian = lambda v: np.matmul(self.jacobian(v), other.evaluate(v)) + np.matmul(self.evaluate(v), other.jacobian(v))
        
        return DifferentiableFunction(
            name=multiplied_function.name,
            domain=multiplied_function.domain,
            evaluate=multiplied_function.evaluate,
            jacobian=jacobian
        )

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
        result = self._jacobian(point)
        if type(result) is np.ndarray:
            return result
        else:
            return np.array([result])

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
    

    ### Aufgabe 4.2: Implemetiere 10 weitere Funktionen: sin, cos, tan, exp, log, sqrt, sigmoid, square, cube, arccos
    @classmethod
    def sin(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a sinus function"""
        sin_function = Function.sin(dimension)
        return cls(name=sin_function.name, domain=sin_function.domain, evaluate=sin_function.evaluate, jacobian=lambda x: np.diag(np.cos(x)))
    
    @classmethod
    def cos(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a cosinus function"""
        cos_function = Function.cos(dimension)
        return cls(name=cos_function.name, domain=cos_function.domain, evaluate=cos_function.evaluate, jacobian=lambda x: np.diag(-np.sin(x)))
    
    @classmethod
    def tan(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a tan function"""
        tan_function = Function.tan(dimension)
        return cls(name=tan_function.name, domain=tan_function.domain, evaluate=tan_function.evaluate, jacobian=lambda x: np.diag(1/np.cos(x)**2))
    
    @classmethod
    def exp(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a exponential function"""
        exp_function = Function.exp(dimension)
        return cls(name=exp_function.name, domain=exp_function.domain, evaluate=exp_function.evaluate, jacobian=lambda x: np.diag(np.exp(x)))
    
    @classmethod
    def log(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a logarithm function"""
        log_function = Function.log(dimension)
        return cls(name=log_function.name, domain=log_function.domain, evaluate=log_function.evaluate, jacobian=lambda x: np.diag(1/x))
    
    @classmethod
    def sqrt(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a square root function"""
        sqrt_function = Function.sqrt(dimension)
        return cls(name=sqrt_function.name, domain=sqrt_function.domain, evaluate=sqrt_function.evaluate, jacobian=lambda x: np.diag(1/(2*np.sqrt(x))))
    
    @classmethod
    def sigmoid(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a sigmoid function"""
        sigmoid_function = Function.sigmoid(dimension)
        return cls(name=sigmoid_function.name, domain=sigmoid_function.domain, evaluate=sigmoid_function.evaluate, jacobian=lambda x: np.diag(np.exp(-x)/(1+np.exp(-x))**2))
    
    @classmethod
    def square(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a square function"""
        square_function = Function.square(dimension)
        return cls(name=square_function.name, domain=square_function.domain, evaluate=square_function.evaluate, jacobian=lambda x: np.diag(2*x))
    
    @classmethod
    def cube(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a cube function"""
        cube_function = Function.cube(dimension)
        return cls(name=cube_function.name, domain=cube_function.domain, evaluate=cube_function.evaluate, jacobian=lambda x: np.diag(3*x**2))
    
    @classmethod
    def arccos(cls, dimension: int) -> IDifferentiableFunction:
        """Returns a arccos function"""
        arccos_function = Function.arccos(dimension)
        return cls(name=arccos_function.name, domain=arccos_function.domain, evaluate=arccos_function.evaluate, jacobian=lambda x: np.diag(-1/np.sqrt(1-x**2)))
    
    ### Ende Aufgabe 4.2
    
    ### Aufgabe 4.3: Implementiere f(x) = (sqrt(cube(x)+2*square(x)-x+1)*exp(sin(square(x))))/(log(square(square(x))+2)+arccos(x/2))
    
    @classmethod
    def own_function(cls) -> IDifferentiableFunction:
        """Returns our own function"""
        dimension = 1
        sqrt_input = DifferentiableFunction(name="x^3+2*x^2-x+1", domain=AffineSpace(dimension), evaluate=lambda x: x**3+2*x**2-x+1, jacobian=lambda x: 3*x**2+4*x-1)
        sqrt_func = DifferentiableFunction.sqrt(dimension)
        sqrt_composed = DifferentiableFunction.FromComposition(sqrt_func, sqrt_input)
        sin_input = DifferentiableFunction.square(dimension)
        sin_func = DifferentiableFunction.sin(dimension)
        sin_composed = DifferentiableFunction.FromComposition(sin_func, sin_input)
        exp_func = DifferentiableFunction.exp(dimension)
        exp_composed = DifferentiableFunction.FromComposition(exp_func, sin_composed)
        zähler = sqrt_composed * exp_composed
        log_input = DifferentiableFunction(name="x^4+2", domain=AffineSpace(dimension), evaluate=lambda x: x**4+2, jacobian=lambda x: 4*x**3)
        log_func = DifferentiableFunction.log(dimension)
        log_composed = DifferentiableFunction.FromComposition(log_func, log_input)
        arccos_input = DifferentiableFunction(name="0.5*x", domain=AffineSpace(dimension), evaluate=lambda x: x/2, jacobian=lambda x: 1/2)
        arccos_func = DifferentiableFunction.arccos(dimension)
        arccos_composed = DifferentiableFunction.FromComposition(arccos_func, arccos_input)
        nenner = (log_composed + arccos_composed)**(-1)
        complete = zähler * nenner
        # print(complete.name)
        return complete
        # return cls(name="own_function", domain=AffineSpace(dimension), evaluate=lambda x: (np.sqrt(x**3+2*x**2-x+1)*np.exp(np.sin(x**2)))/(np.log(x**4+2)+np.arccos(x/2)))