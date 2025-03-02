import numpy as np
from DifferentiableFunction import IDifferentiableFunction, DifferentiableFunction
from Set import AffineSpace
from typing import Callable

class GradientDescent(object):

    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, startingpoint: np.array, iterations: int = 100, learningrate: float = 0.1) -> np.array:
        x = startingpoint
        for step in range(iterations):
            gradient = function.jacobian(x).reshape([-1])
            x = x - learningrate * gradient
        return x

    def StochasticMinimize(self, toLoss: Callable, data_x: np.array, data_y: np.array, startingpoint: np.array, iterations: int = 100, learningrate: float = 0.1, batch_size: int = 1) -> np.array:
        """
        Stochastic gradient descent
        :param toLoss: A function that takes x and y as input and returns a loss function
        :param data_x: The input data
        :param data_y: The output data
        :param startingpoint: The starting point
        :param iterations: The number of iterations
        :param learningrate: The learning rate
        :param batch_size: The batch size
        :return: The optimized parameters
        Complexity: O(iterations*n/batch_size * d * h^2), 
        where n is the number of data points, d is the dimension of the data points, h is the hidden dimension
        """
        x = startingpoint
        n = data_x.shape[0]
        
        for step in range(iterations):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(n)
            np.random.shuffle(indices)
            
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch_indices = indices[batch_start:batch_end]
                
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                
                # Compute the gradient for the mini-batch
                # Complexity of toLoss: # Complexity: O(n * d * h^2), 
                # where n is the number of data points, d is the dimension of the data points, h is the hidden dimension
                loss = toLoss(batch_x, batch_y)
                gradient = loss.jacobian(x).reshape([-1])
                x = x - learningrate * gradient

        return x