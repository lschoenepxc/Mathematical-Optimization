import numpy as np
import time
import tracemalloc
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from BFGS import BFGS
import matplotlib.pyplot as plt

def profile_bfgs(dimension, iterations=100):
    R = AffineSpace(dimension)
    f = DifferentiableFunction(
        name="x->sum(x^2)", domain=R,
        evaluate=lambda x: np.array([np.sum(x**2)]),
        jacobian=lambda x: 2 * x.reshape(1, -1))
    
    bfgs = BFGS()
    starting_point = np.random.uniform(-10, 10, dimension)
    
    # Start Profiling
    tracemalloc.start()
    start_time = time.time()
    
    result = bfgs.Minimize(f, startingpoint=starting_point, iterations=iterations)
    
    # End Profiling
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    memory_usage = peak / 10**6  # in MB
    
    return elapsed_time, memory_usage

# Test für verschiedene Dimensionen
dimensions = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
results = []

for dim in dimensions:
    elapsed_time, memory_usage = profile_bfgs(dim)
    results.append((dim, elapsed_time, memory_usage))
    print(f"Dimension: {dim}, Time: {elapsed_time:.4f} seconds, Memory: {memory_usage:.4f} MB")

"""
Laufzeitkomplexität
    Der BFGS-Algorithmus besteht aus mehreren Schritten, die iterativ ausgeführt werden:

    Gradientenberechnung: Die Berechnung des Gradienten hat eine Laufzeit von (O(n)), wobei (n) die Dimension des Eingaberaums ist.
    Matrix-Vektor-Multiplikation: Die Multiplikation der Hesse-Matrix (appr.) mit dem Gradienten hat eine Laufzeit von (O(n^2)).
    Line Search: Die Line Search hat eine variable Laufzeit, die von der Variante abhängt.
    Hesse-Matrix-Update: Das Update der Hesse-Matrix hat eine Laufzeit von (O(n^2)).
    Da diese Schritte in jeder Iteration ausgeführt werden, beträgt die Gesamtlaufzeit pro Iteration (O(n^2)). 
    Wenn der Algorithmus (k) Iterationen benötigt, beträgt die Gesamtlaufzeit (O(kn^2)).

Speicherkomplexität
    Der Speicherverbrauch des BFGS-Algorithmus wird hauptsächlich durch die Speicherung der Hesse-Matrix und der Gradienten bestimmt:

    Hesse-Matrix: Die Hesse-Matrix hat eine Größe von (n x n), was einen Speicherverbrauch von (O(n^2)) bedeutet.
    Gradient: Der Gradient hat eine Größe von (n), was einen Speicherverbrauch von (O(n)) bedeutet.
    Der Gesamtspeicherverbrauch beträgt somit (O(n^2)).
"""

"""
Ergebnisse des Profilings:
Dimension: 10, Time: 0.0034 seconds, Memory: 0.0108 MB
Dimension: 50, Time: 0.0030 seconds, Memory: 0.1269 MB
Dimension: 100, Time: 0.0039 seconds, Memory: 0.4611 MB
Dimension: 200, Time: 0.0067 seconds, Memory: 1.6158 MB
Dimension: 500, Time: 0.0299 seconds, Memory: 10.0351 MB
Dimension: 1000, Time: 0.2181 seconds, Memory: 40.0671 MB
"""
"""
Dimension: 10, Time: 0.0031 seconds, Memory: 0.0111 MB
Dimension: 50, Time: 0.0044 seconds, Memory: 0.1269 MB
Dimension: 100, Time: 0.0037 seconds, Memory: 0.4611 MB
Dimension: 200, Time: 0.0069 seconds, Memory: 1.6158 MB
Dimension: 500, Time: 0.0283 seconds, Memory: 10.0351 MB
Dimension: 1000, Time: 0.1911 seconds, Memory: 40.0671 MB
Dimension: 2000, Time: 1.1516 seconds, Memory: 160.1310 MB
Dimension: 5000, Time: 15.9269 seconds, Memory: 1000.3230 MB
Dimension: 10000, Time: 135.6213 seconds, Memory: 4000.6430 MB
"""


# Plotting
# dimensions, times, memory = zip(*results)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(dimensions, times, marker='o')
# plt.xlabel('Dimension')
# plt.ylabel('Time (s)')
# plt.title('BFGS Runtime vs. Dimension')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(dimensions, memory, marker='o')
# plt.xlabel('Dimension')
# plt.ylabel('Memory (MB)')
# plt.title('BFGS Memory Usage vs. Dimension')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# --> Result in profile_BFGS.png

"""
Laufzeitwerte:
    Laufzeit steigt mit der Dimension, da die Anzahl der Berechnungen quadratisch mit der Dimension zunimmt.
    Für Dimensionen von 10 bis 10000 beträgt die Laufzeit etwa 0.0031 bis 135.6213 Sekunden.
    Die Laufzeit für 10000 Dimensionen ist etwa 43839-mal höher als für 10 Dimensionen.
Speicherwerte:
    Speicherverbrauch steigt mit der Dimension, da die Hesse-Matrix quadratisch mit der Dimension wächst.
    Für Dimensionen von 10 bis 10000 beträgt der Speicherverbrauch etwa 0.0111 bis 4000.6430 MB.
    Der Speicherverbrauch für 10000 Dimensionen ist etwa 360.000-mal höher als für 10 Dimensionen.
Zusammenfassung:
    Der BFGS-Algorithmus hat eine quadratische Laufzeit und einen quadratischen Speicherverbrauch in Bezug auf die Dimension des Eingaberaums.
    Dies bedeutet, dass die Laufzeit und der Speicherverbrauch exponentiell mit der Dimension zunehmen.
    Für hochdimensionale Probleme kann der BFGS-Algorithmus aufgrund seines hohen Ressourcenbedarfs ineffizient sein.
    Es ist wichtig, die Dimension des Problems zu berücksichtigen und gegebenenfalls alternative Algorithmen zu verwenden, 
    die besser mit hochdimensionalen Problemen umgehen können.
"""

