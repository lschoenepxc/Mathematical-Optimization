import numpy as np
import math
from typing import Callable, Optional
from DifferentiableFunction import IDifferentiableFunction
from Set import AffineSpace, MultidimensionalInterval
from GP import GP
from SQP import SQP


class BO(object):
    def __init__(self):
        super().__init__()

    def Minimize(self, function: IDifferentiableFunction, iterations: int = 50, x: Optional[np.array] = None, y: Optional[np.array] = None) -> np.array:

        domain = function.domain
        d = domain._ambient_dimension

        # Initialize data_x and data_y with provided data or empty arrays
        if x is not None and y is not None:
            # check that x and y is not empty
            assert x.shape[0] > 0, "The data_x must not be empty."
            assert y.shape[0] > 0, "The data_y must not be empty."
            # data_x dimension must be equal to the domain dimension
            assert x.shape[1] == d, "The data_x must have the same dimension as the domain."
            # check that the number of rows in x and y is equal
            assert x.shape[0] == y.shape[0], "The number of rows in data_x and data_y must be equal."
            assert domain.contains(x), "The data_x must be in the domain of the function."
            data_x = x
            data_y = y
        else:
            data_x = np.empty((0, d))
            data_y = np.empty((0,))
        
        gp = GP(data_x=data_x, data_y=data_y)
        # gp = GP(data_x=data_x, data_y=data_y, kernel=GP.MaternCovariance(nu=1.5, length_scale=1))
        sqp = SQP()

        for step in range(iterations):
            # UCB with 2 sigma, added with 0 times the original function to get the domain
            acquisition_function = gp.PosteriorMean(
            )-2*gp.PosteriorStandardDeviation() + 0*function

            # new measurement point
            startingpoint = domain.point()
            x = sqp.Minimize(acquisition_function,
                             startingpoint=startingpoint)
            # measure
            y = function.evaluate(x)

            # update data
            data_x = np.concatenate((data_x, x.reshape(1, -1)), axis=0)
            data_y = np.concatenate((data_y, y), axis=0)
            gp = GP(data_x=data_x, data_y=data_y)

        return data_x[np.argmin(data_y), :]
