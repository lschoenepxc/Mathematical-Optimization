from Set import ISet, RealNumbers
from abc import abstractmethod, ABC
from typing import List, Union

class IField(ABC):
    def __init__(self, set: ISet):
        super().__init__()
        self._set = set
        
    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def mul(self, a, b):
        pass

    @abstractmethod
    def sub(self, a, b):
        pass

    @abstractmethod
    def div(self, a, b):
        pass

    @abstractmethod
    def neg(self, a):
        pass

    @abstractmethod
    def inv(self, a):
        pass

    @abstractmethod
    def zero(self):
        pass

    @abstractmethod
    def one(self):
        pass
    
class Field(IField):
    def __init__(self, set: ISet):
        super().__init__(set)
    
    def _check_element(self, element):
        if not self._set.contains(element):
            raise ValueError(f"Element {element} is not in the set {self._set}")

    def add(self, a, b):
        self._check_element(a)
        self._check_element(b)
        return a + b
    
    def mul(self, a, b):
        self._check_element(a)
        self._check_element(b)
        return a * b
    
    def sub(self, a, b):
        self._check_element(a)
        self._check_element(b)
        return a - b
    
    def div(self, a, b):
        self._check_element(a)
        self._check_element(b)
        if b == 0:
            raise ZeroDivisionError("Division durch 0 ist nicht erlaubt.")
        return a / b
    
    def neg(self, a):
        self._check_element(a)
        return -a
    
    def inv(self, a):
        self._check_element(a)
        if a == 0:
            raise ZeroDivisionError("0 hat kein multiplikatives Inverses.")
        return 1/a
    
    def zero(self):
        return 0
    
    def one(self):
        return 1
    
class IVectorSpace(ABC):
    """Interface für einen Vektorraum."""
    
    def __init__(self, field: Field):
        self._field = field
    
    @abstractmethod
    def add(self, v: List[float], w: List[float]) -> List[float]:
        """Vektoraddition."""
        pass
    
    @abstractmethod
    def scalar_mul(self, alpha: float, v: List[float]) -> List[float]:
        """Multiplikation mit einem Skalar."""
        pass
    
    @abstractmethod
    def zero_vector(self, dim: int) -> List[float]:
        """Nullvektor der gegebenen Dimension."""
        pass

class VectorSpace(IVectorSpace):
    """Konkrete Implementierung eines Vektorraums über einem gegebenen Körper."""
    
    def add(self, v: List[float], w: List[float]) -> List[float]:
        if len(v) != len(w):
            raise ValueError("Vektoren müssen die gleiche Dimension haben.")
        return [self._field.add(v[i], w[i]) for i in range(len(v))]
    
    def scalar_mul(self, alpha: float, v: List[float]) -> List[float]:
        return [self._field.mul(alpha, v[i]) for i in range(len(v))]
    
    def zero_vector(self, dim: int) -> List[float]:
        return [self._field.zero()] * dim
    
class ILinearMap(ABC):
    """Interface für eine lineare Abbildung zwischen zwei Vektorräumen."""
    
    def __init__(self, domain: VectorSpace, codomain: VectorSpace):
        self._domain = domain
        self._codomain = codomain
    
    @abstractmethod
    def apply(self, v: Union[List[Union[float, int]], List[List[Union[float, int]]]]) -> Union[List[float], List[List[float]]]:
        """Wendet die lineare Abbildung auf einen Vektor oder eine Matrix an."""
        pass
    
    @abstractmethod
    def matrix_representation(self) -> List[List[float]]:
        """Gibt die Matrixdarstellung der Abbildung zurück."""
        pass

class LinearMap(ILinearMap):
    """Konkrete Implementierung einer linearen Abbildung als Matrixmultiplikation."""
    
    def __init__(self, domain: VectorSpace, codomain: VectorSpace, matrix: List[List[Union[float, int]]]):
        super().__init__(domain, codomain)
        self._matrix = matrix
    
    def apply(self, v: Union[List[Union[float, int]], List[List[Union[float, int]]]]) -> Union[List[float], List[List[float]]]:
        if isinstance(v, list) and isinstance(v[0], (float, int)):  # Vektor
            if len(v) != len(self._matrix[0]):
                raise ValueError("Vektor-Dimension muss mit Spaltenanzahl der Matrix übereinstimmen.")
            return [
                sum(self._domain._field.mul(self._matrix[i][j], float(v[j])) for j in range(len(v)))
                for i in range(len(self._matrix))
            ]
        elif isinstance(v, list) and isinstance(v[0], list):  # Matrix
            if len(v) != len(self._matrix[0]):
                raise ValueError("Matrix-Spaltenanzahl muss mit Zeilenanzahl der zweiten Matrix übereinstimmen.")
            return [
                [
                    sum(self._domain._field.mul(self._matrix[i][k], float(v[k][j])) for k in range(len(v)))
                    for j in range(len(v[0]))
                ]
                for i in range(len(self._matrix))
            ]
        else:
            raise ValueError("Eingabe muss ein Vektor oder eine Matrix sein.")
    
    def matrix_representation(self) -> List[List[float]]:
        return self._matrix
    


    
