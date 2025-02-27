import numpy as np
import time
from GP import GP

def measure_alpha_time(data_x, data_y, use_cho, iterations=10):
    times = []
    for _ in range(iterations):
        # Für jeden Testlauf wird ein frischer GP erzeugt,
        # sodass keine gecachten Ergebnisse übernommen werden.
        gp = GP(data_x=data_x, data_y=data_y)
        GP.reset_cache(gp)
        start = time.time()
        # __alpha ist als private Methode durch Namens-Mangling erreichbar:
        _ = gp._GP__alpha(use_cho=use_cho)
        end = time.time()
        times.append(end - start)
    return sum(times) / iterations

def test_variants():
    sizes = [50, 100, 200, 500]
    d = 1  # Beispiel: 1-dimensionale Eingaben

    print("n, cho_solve (s), np.linalg.solve (s)")
    for n in sizes:
        data_x = np.random.rand(n, d)
        data_y = np.random.rand(n)

        t_cho = measure_alpha_time(data_x, data_y, use_cho=True)
        t_solve = measure_alpha_time(data_x, data_y, use_cho=False)

        print(f"{n}, {t_cho:.6f}, {t_solve:.6f}")

if __name__ == "__main__":
    test_variants()