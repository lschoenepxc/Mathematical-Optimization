from Set import RealNumbers, MultidimensionalInterval
from Field import RealNumberField, VectorSpace, LinearMap
import numpy as np

# Beispiel-Feld
real_numbers = RealNumbers()
field = RealNumberField(real_numbers)

# Operationen auf den Elementen des Sets
a = 3.0
b = 4.0

print("Add:", field.add(a, b))
print("Mul:", field.mul(a, b))
print("Sub:", field.sub(a, b))
print("Div:", field.div(a, b))
print("Neg:", field.neg(a))
print("Inv:", field.inv(a))
print("Zero:", field.zero())
print("One:", field.one())

# Testen mit einem Element, das nicht im Set enthalten ist
try:
    c = "not a number"
    print("Add with invalid element:", field.add(a, c))
except ValueError as e:
    print(e)
    
# Vektorraum über den reellen Zahlen
vector_space = VectorSpace(field)

# Beispiel-Vektoren
v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]

print("Vektoraddition:", vector_space.add(v, w))
print("Skalarmultiplikation:", vector_space.scalar_mul(2.0, v))
print("Nullvektor:", vector_space.zero_vector(len(v)))

# Lineare Abbildung als Matrix (3x3)
matrix = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
linear_map = LinearMap(vector_space, vector_space, matrix)

# Beispiel-Vektor
v = [1, 2, 3]

print("Bild des Vektors unter der Abbildung:", linear_map.apply(v))
print("Matrixdarstellung der Abbildung:", linear_map.matrix_representation())

# Matrix-Matrix-Multiplikation
A = LinearMap(vector_space, vector_space, [[1, 2], [3, 4]])
v = [5, 6]
B = [[7, 8], [9, 10]]

print("Matrix * Vektor:", A.apply(v))  # Matrix-Vektor-Multiplikation
print("Matrix * Matrix:", A.apply(B))  # Matrix-Matrix-Multiplikation
