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
                loss = toLoss(batch_x, batch_y)
                gradient = loss.jacobian(x).reshape([-1])
                x = x - learningrate * gradient

        return x