import unittest
import itertools
import numpy as np
from GP import GP


class tests_GP(unittest.TestCase):

    def test_GP_RBF(self):

        data_x = np.array([[1, 2], [3, 4], [5, 6]])
        data_y = np.array([7, 8, 9])
        gp = GP(data_x=data_x, data_y=data_y, kernel=GP.RBF())
        # == GP(data_x=data_x, data_y=data_y), because RBF is the default kernel
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([1, 2])).item(), 7, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([3, 4])).item(), 8, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([5, 6])).item(), 9, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([1, 2])).item(), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([3, 4])).item(), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([5, 6])).item(), 0, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([1, 2])).item(), 0, 2)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([3, 4])).item(), 0, 2)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([5, 6])).item(), 0, 2)

        self.assertAlmostEqual(mu.evaluate(
            np.array([0, 0])).item(), 0.5630289394, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([2, 1])).item(), 2.575156089, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([4, 3])).item(), 2.943035531, 3)
        self.assertAlmostEqual(mu.evaluate(
            np.array([6, 5])).item(), 3.310914971, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([0, 0])) -
                                              np.array([0.5630864202, 1.126115360])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([2, 1])) -
                                              np.array([-2.471226856, 2.679085323])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([4, 3])) -
                                              np.array([-2.916083740, 2.969987321])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(mu.jacobian(np.array([6, 5])) -
                                              np.array([-3.414843874, 3.206986069])), 0, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([0, 0])).item(), 0.993259802354344, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([2, 1])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([4, 3])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([6, 5])).item(), 0.864664716742376, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([0, 0])) -
                                              np.array([-0.0134803729206716, -0.0269607682119837])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([2, 1])) -
                                              np.array([0.270670566515277, -0.270670566515220])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([4, 3])) -
                                              np.array([0.270670566515249, -0.270670566515249])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma2.jacobian(np.array([6, 5])) -
                                              np.array([0.270670566515220, -0.270670566515278])), 0, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([0, 0])).item(), 0.996624203175070, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([2, 1])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([4, 3])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([6, 5])).item(), 0.929873495020896, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([0, 0])) -
                                              np.array([-0.00676301703175857, -0.0135260452867246])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([2, 1])) -
                                              np.array([0.145541607522212, -0.145541607522181])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([4, 3])) -
                                              np.array([0.145541607522196, -0.145541607522196])), 0, 3)
        self.assertAlmostEqual(np.linalg.norm(sigma.jacobian(np.array([6, 5])) -
                                              np.array([0.145541607522181, -0.145541607522212])), 0, 3)

    def test_GP_trivial_RBF(self):

        data_x = np.empty((0, 0))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y)
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)

    def test_GP_trivial2_RBF(self):

        data_x = np.empty((0, 3))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y)
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)
        
        
    def test_GP_trivial_Matern(self):

        data_x = np.empty((0, 0))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y, kernel=GP.MaternCovariance())
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)

    def test_GP_trivial2_Matern(self):

        data_x = np.empty((0, 3))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y, kernel=GP.MaternCovariance())
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        self.assertAlmostEqual(mu.evaluate(
            np.array([])).item(), 0)
        self.assertAlmostEqual(sigma2.evaluate(
            np.array([])).item(), 1)
        self.assertAlmostEqual(sigma.evaluate(
            np.array([])).item(), 1)
        
    def test_GP_trivial_Linear(self):
        # Leere Daten
        data_x = np.empty((0, 0))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y, kernel=GP.Linear())
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        # Überprüfen Sie, ob der Mittelwert 0 ist
        self.assertAlmostEqual(mu.evaluate(np.array([])).item(), 0)
        # Überprüfen Sie, ob die Varianz 0 ist (da der lineare Kernel keine Varianz hinzufügt)
        self.assertAlmostEqual(sigma2.evaluate(np.array([])).item(), 0)
        # Überprüfen Sie, ob die Standardabweichung 0 ist
        self.assertAlmostEqual(sigma.evaluate(np.array([])).item(), 0)

    def test_GP_trivial2_Linear(self):
        # Leere Daten mit 3-dimensionalem Eingaberaum
        data_x = np.empty((0, 3))
        data_y = np.empty((0,))
        gp = GP(data_x=data_x, data_y=data_y, kernel=GP.Linear())
        mu = gp.PosteriorMean()
        sigma2 = gp.PosteriorVariance()
        sigma = gp.PosteriorStandardDeviation()

        # Überprüfen Sie, ob der Mittelwert 0 ist
        self.assertAlmostEqual(mu.evaluate(np.array([])).item(), 0)
        # Überprüfen Sie, ob die Varianz 0 ist (da der lineare Kernel keine Varianz hinzufügt)
        self.assertAlmostEqual(sigma2.evaluate(np.array([])).item(), 0)
        # Überprüfen Sie, ob die Standardabweichung 0 ist
        self.assertAlmostEqual(sigma.evaluate(np.array([])).item(), 0)

        
    def test_GP_kernel_comparison_linear_data(self):
        # Lineare Daten
        data_x = np.array([[1, 2], [3, 4], [5, 6]])
        data_y = np.array([5, 11, 17])  # y = 2*x1 + x2 + 1

        # Verschiedene Kerne
        kernels = {
            "RBF": GP.RBF(),
            "Matern": GP.MaternCovariance(),
            "Linear": GP.Linear()
        }

        results = {}

        for name, kernel in kernels.items():
            gp = GP(data_x=data_x, data_y=data_y, kernel=kernel)
            mu = gp.PosteriorMean()
            sigma2 = gp.PosteriorVariance()
            sigma = gp.PosteriorStandardDeviation()

            # Speichern der Ergebnisse
            results[name] = {
                "mu": mu.evaluate(np.array([1, 2])).item(),
                "sigma2": sigma2.evaluate(np.array([1, 2])).item(),
                "sigma": sigma.evaluate(np.array([1, 2])).item()
            }
        # print("Linear Data")
        # # Vergleiche der Ergebnisse
        # for name, result in results.items():
        #     print(f"Kernel: {name}")
        #     print(f"Posterior Mean: {result['mu']}")
        #     print(f"Posterior Variance: {result['sigma2']}")
        #     print(f"Posterior Standard Deviation: {result['sigma']}")
        #     print()
        
    def test_GP_kernel_comparison_noisy_exponential_data(self):
        # Verrauschte exponentielle Daten
        np.random.seed(0)
        data_x = np.array([[1, 2], [3, 4], [5, 6]])
        data_y = np.exp(data_x[:, 0]) + np.exp(data_x[:, 1]) + np.random.normal(0, 0.1, data_x.shape[0])

        # Verschiedene Kerne
        kernels = {
            "RBF": GP.RBF(),
            "Matern": GP.MaternCovariance(),
            "Linear": GP.Linear()
        }

        results = {}

        for name, kernel in kernels.items():
            gp = GP(data_x=data_x, data_y=data_y, kernel=kernel)
            mu = gp.PosteriorMean()
            sigma2 = gp.PosteriorVariance()
            sigma = gp.PosteriorStandardDeviation()

            # Speichern der Ergebnisse
            results[name] = {
                "mu": mu.evaluate(np.array([1, 2])).item(),
                "sigma2": sigma2.evaluate(np.array([1, 2])).item(),
                "sigma": sigma.evaluate(np.array([1, 2])).item()
            }

        # print("Noisy Exponential Data")
        # # Vergleiche der Ergebnisse
        # for name, result in results.items():
        #     print(f"Kernel: {name}")
        #     print(f"Posterior Mean: {result['mu']}")
        #     print(f"Posterior Variance: {result['sigma2']}")
        #     print(f"Posterior Standard Deviation: {result['sigma']}")
        #     print()

if __name__ == '__main__':
    unittest.main()
