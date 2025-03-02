from Set import ISet, RealNumbers
from abc import abstractmethod, ABC
from typing import List, Union, Optional

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

    # additive Inverse
    @abstractmethod
    def neg(self, a):
        pass
    
    # multiplikatives Inverse
    @abstractmethod
    def inv(self, a):
        pass

    @abstractmethod
    def zero(self):
        pass

    @abstractmethod
    def one(self):
        pass
    
class RealNumberField(IField):
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
    
    def __init__(self, field: IField):
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
        """
        Wendet die lineare Abbildung auf einen Vektor oder eine Matrix an.
        :param v: Vektor oder Matrix
        :return: Vektor oder Matrix
        """
        if isinstance(v, list) and isinstance(v[0], (float, int)):  # Vektor
            if len(v) != len(self._matrix[0]):
                raise ValueError("Vektor-Dimension muss mit Spaltenanzahl der Matrix übereinstimmen.")
            return [
                sum(self._domain._field.mul(self._matrix[i][j], v[j]) for j in range(len(v)))
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
    
class gauss_jordan_algorithm():
    """Solves a linear system of equations using the Gauss algorithm."""
    def __init__(self, matrix: List[List[float]], b: Optional[List[float]] = None):
        self._matrix = matrix
        self._b = b
        assert all(len(row) == len(matrix[0]) for row in matrix), "Matrix must have the same number of columns in each row."
        assert len(matrix) == len(matrix[0]), "Matrix must be quadratic."
        self._n = len(matrix)
        if self._b is not None:
            assert len(self._b) == self._n, "Number of rows in b must be equal to the number of rows in the matrix."
        self._m = len(matrix[0])
        self._domain = VectorSpace(RealNumberField(RealNumbers()))
        self._operations = []  # Liste zur Speicherung der Operationen
        
    # Elementarmatrizen als lineare Abbildungen (Zeilenvertauschung, Zeilenskalierung, Zeilenaddition)
    def swap_rows(self, i: int, j: int) -> LinearMap:
        matrix = [[1 if k == l else 0 for l in range(self._n)] for k in range(self._n)]
        matrix[i][i] = 0
        matrix[j][j] = 0
        matrix[i][j] = 1
        matrix[j][i] = 1
        return LinearMap(self._domain, self._domain, matrix)
    
    def scale_row(self, i: int, alpha: float) -> LinearMap:
        matrix = [[1 if k == l else 0 for l in range(self._n)] for k in range(self._n)]
        matrix[i][i] = alpha
        return LinearMap(self._domain, self._domain, matrix)
    
    def add_row(self, i: int, j: int, alpha: float) -> LinearMap:
        matrix = [[1 if k == l else 0 for l in range(self._n)] for k in range(self._n)]
        matrix[i][j] = alpha
        return LinearMap(self._domain, self._domain, matrix)
    
    def solve(self) -> List[float]:
        if self._b is not None:
            self._matrix = [self._matrix[i] + [self._b[i]] for i in range(self._n)]
            
        # Erzeuge Einheitsmatrix
        matrix = [[1 if i == j else 0 for j in range(self._n)] for i in range(self._n)]
        # Führe Gauss-Algorithmus durch
        for i in range(self._n):
            # Spalte i nach unten durchgehen
            for j in range(i, self._n):
                if self._matrix[j][i] != 0:
                    # Tausche Zeilen i und j
                    swap_op = self.swap_rows(i, j)
                    matrix = swap_op.apply(matrix)
                    self._matrix = swap_op.apply(self._matrix)
                    self._operations.append(swap_op)
                    break
            # Skaliere Zeile i
            scale_op = self.scale_row(i, 1/self._matrix[i][i])
            matrix = scale_op.apply(matrix)
            self._matrix = scale_op.apply(self._matrix)
            self._operations.append(scale_op)
            # Subtrahiere Vielfaches von Zeile i von den anderen Zeilen
            for j in range(self._n):
                if j != i:
                    add_op = self.add_row(j, i, -self._matrix[j][i])
                    matrix = add_op.apply(matrix)
                    self._matrix = add_op.apply(self._matrix)
                    self._operations.append(add_op)
        
        # Extrahiere die Lösung aus der erweiterten Matrix
        solution = [self._matrix[i][-1] for i in range(self._n)]
        
        return solution
    
    def get_operations(self) -> List[LinearMap]:
        """Gibt die Liste der gespeicherten Operationen zurück."""
        return self._operations
    
    def compute_inverse(self) -> List[List[float]]:
        """Berechnet die Inverse der Matrix basierend auf den gespeicherten Operationen."""
        inverse = [[1 if i == j else 0 for j in range(self._n)] for i in range(self._n)]
        for op in self._operations:
            inverse = op.apply(inverse)
        return inverse