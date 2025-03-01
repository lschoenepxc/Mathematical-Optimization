import numpy as np
import scipy as sp
import math
from DifferentiableFunction import DifferentiableFunction
from Set import AffineSpace
from typing import Callable
from functools import lru_cache

class GP(object):
    def __init__(self, data_x: np.array, data_y: np.array, kernel: Callable[[np.ndarray], np.ndarray] = None):
        super().__init__()
        assert data_x.shape[0] == data_y.shape[0], "need as many x-positions as y-positions."
        self.data_x = data_x
        self.data_y = data_y
        self.n = self.data_x.shape[0]
        self.d = self.data_x.shape[1]
        # self.kernel = lambda x1, x2: math.exp(-0.5*np.linalg.norm(x1-x2)**2)
        if kernel is None:
            kernel = self.RBF()
        self.kernel = kernel

    def __K(self) -> np.array:
        """The covariance matrix of this GP"""
        if not hasattr(self, 'K'):
            self.K = np.array([[self.kernel(self.data_x[i, :], self.data_x[j, :]) + (i == j) * 1e-5
                                for i in range(self.n)] for j in range(self.n)]).reshape(self.n, self.n)
        return self.K

    def __L(self) -> np.array:
        """The Cholesky factor of the covariance matrix K (notation as in Rasmussen&Williams) of this GP"""
        if not hasattr(self, 'L'):
            self.L = np.linalg.cholesky(self.__K())
        return self.L

    @lru_cache(maxsize=None)
    def _cached_kernel_impl(self, x1_tuple, x2_tuple):
        """Diese Funktion arbeitet NUR mit Tupeln. Dadurch ist sie lru_cache-kompatibel."""
        # Aus den Tupeln wieder ein numpy-Array bauen
        x1 = np.array(x1_tuple)
        x2 = np.array(x2_tuple)
        return self.kernel(x1, x2)

    def cached_kernel(self, x1, x2):
        """Public-Methode, die x1 und x2 (Arrays) zu Tupeln macht und dann _cached_kernel_impl aufruft."""
        return self._cached_kernel_impl(tuple(x1), tuple(x2))

    def __alpha(self, use_cho=True) -> np.array:
        """The vector alpha (notation as in Rasmussen&Williams) of this GP"""
        if not hasattr(self, 'alpha'):
            # this is numerically less stable than using the cholesky decomposition
            # self.alpha = np.linalg.solve(self.__K(), self.data_y)
            # self.alpha = np.linalg.solve(self.__L().T, np.linalg.solve(self.__L(), self.data_y))
            ### Comments todo:
            # Effizienz: Laufzeit, Speicher, O-Notation
            # wartbarkeit: Lesbarkeit, Verständlichkeit, Kommentare, Tests!!! --> tatsächlich numerisch stabiler?
            # Lesbarkeit: Variablen, Funktionen, Kommentare
            # scipy cho_solve --> viiiel zu langsam!! sollte aber nicht so sein: 
            # https://stackoverflow.com/questions/66382370/performance-gap-between-np-linalg-solve-and-scipy-linalg-cho-solve
            if use_cho:
                self.alpha = sp.linalg.cho_solve((self.__L(), True), self.data_y, check_finite=False)
            else:
            # self.alpha = self._choleskySolve((self.__L()), self.data_y)
            # wothout cholesky solve
                z = np.linalg.solve(self.__L(), self.data_y)
                self.alpha = np.linalg.solve(self.L.T, z)
        return self.alpha
    
    def reset_cache(self):
        """Löscht alle in der Instanz gecachten Daten."""
        for attr in ['K', 'L', 'alpha']:
            if hasattr(self, attr):
                delattr(self, attr)
        # lru_cache des Kernel-Implementierung leeren
        self._cached_kernel_impl.cache_clear()

    def __ks(self, x: np.array) -> np.array:
        """The vector k_*=k(x_*,X) (notation as in Rasmussen&Williams) given of this GP"""
        return np.array([self.kernel(x, self.data_x[i, :]) for i in range(self.n)])

    def __dks(self, x: np.array) -> np.array:
        """The derivative of the vector k_*=k(x_*,X) (notation as in Rasmussen&Williams) given of this GP"""
        return np.array([list(self.kernel(x, self.data_x[i, :])*(self.data_x[i, :]-x)) for i in range(self.n)])
    
    # needed for scipy from version 1.15.1 onward (not needed for 1.14)
    def _choleskySolve(self, A:np.ndarray, b:np.ndarray) -> np.ndarray:

        # the assertion could be useful, BUT this is a private function, called three times by me
        #assert np.allclose(A, np.tril(A)), "choleskySolve expects a lower triangular Matrix as A" 
        try:
            result = sp.linalg.cho_solve((A, True), b, check_finite=False) 
        except Exception as ex:
            # scipy throws an error in trivial test cases, in this case numpy yields the 'correct' result
            result = np.linalg.solve(A, np.linalg.solve(np.transpose(A), b))
        return result

    # calculate posterior mean, variance and standard deviation with cholesky decomposition
    def PosteriorMean(self):
        return DifferentiableFunction(
            name="GP_posterior_mean",
            domain=AffineSpace(self.d),
            evaluate=lambda x: np.dot(self.__alpha(), self.__ks(x)),
            jacobian=lambda x: np.dot(self.__alpha(), self.__dks(x))
        )
        
    # Posterior Kovarianz: (x,x') -> k(x,x') - alpha^T * k(x,X) - k(x',X)
    def PosteriorVariance(self):
        return DifferentiableFunction(
            name="GP_posterior_variance",
            domain=AffineSpace(self.d),
            # evaluate=lambda x: np.array([self.kernel(x, x)-np.linalg.norm(np.linalg.solve(self.__L(), self.__ks(x)))**2]),
            evaluate=lambda x: np.array([self.kernel(x, x)-np.linalg.norm(sp.linalg.solve_triangular(self.__L(), self.__ks(x), lower=True))**2]),
            # L from cholesky decomposition of K already calculated in __L --> no cholesky decomposition needed here
            # jacobian=lambda x: 0-2 * np.reshape(np.dot(np.linalg.solve(self.__L(), self.__ks(x)), np.linalg.solve(self.__L(), self.__dks(x))), (1, -1))
            jacobian=lambda x: 0-2 * np.reshape(np.dot(sp.linalg.solve_triangular(self.__L(), self.__ks(x), lower=True), sp.linalg.solve_triangular(self.__L(), self.__dks(x), lower=True)), (1, -1))
        )

    def PosteriorStandardDeviation(self):
        sqrt = DifferentiableFunction(name="sqrt", domain=AffineSpace(
            1), evaluate=lambda x: np.sqrt(x), jacobian=lambda x: np.reshape(0.5/np.sqrt(x), (1, 1)))
        return DifferentiableFunction.FromComposition(sqrt, self.PosteriorVariance())
    
    @staticmethod
    def MaternCovariance(nu: float=2.5, length_scale: float = 1.0, sigma: float = 1.0):
        def matern_kernel(x1, x2):
            distance = np.linalg.norm(x1 - x2)
            factor = np.sqrt(2 * nu) * distance / length_scale
            if factor == 0.0:
                return 1.0
            else:
                return sigma**2 * (2 ** (1 - nu) / math.gamma(nu)) * (factor ** nu) * sp.special.kv(nu, factor)
        return matern_kernel
    
    @staticmethod
    def RBF(sigma: float = 1.0, length_scale: float = 1.0):
        # Standard with sigma = 1 and length_scale = 1 --> Gaussian Kernel
        rbf_kernel = lambda x, xp: sigma**2 * np.exp(-np.linalg.norm(x - xp)**2 / (2 * length_scale**2))
        return rbf_kernel
    
    @staticmethod
    def Linear(sigma: float = 1.0):
        linear_kernel = lambda x, xp: sigma**2 * np.dot(x, xp)
        return linear_kernel
