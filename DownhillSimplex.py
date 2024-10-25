import numpy as np
from Function import IFunction, Function
from Set import AffineSpace, MultidimensionalInterval

class DownhillSimplex(object):

    def __init__(self):
        super().__init__()

    def checkParams(params: dict) -> bool:
        # Check if all necessary parameters are within bounds
        # alpha > 0, gamma > 0, beta in (0,1), delta in (0,1)
        lowerBound = 0.0
        upperBounds = [float('inf'),float('inf'),1.0,1.0]
        
        return all([lowerBound < params[key] < upperBounds[i] for i, key in enumerate(params.keys())])
        
    # Schleifeninvarianz!!
    def checkLinearIndependency(points: np.array) -> bool:
        # Check if for all points x_i but x_0: x_i-x_0 linearly independent 
        # (points/x_0).shape[0] == points.shape[1] --> quadratic matrix
        matrix = [points[i] - points[0] for i in range(1, points.shape[0])]
        return np.linalg.det(matrix) != 0
        
    def sortPoints(x: np.array,y: np.array) -> np.array:
        # Sort points by their function value
        return x[np.argsort(y)]
    
    def evalMin(x1: np.array, x2: np.array, function: IFunction) -> np.array:
        # Set the minimum of two points
        return x1 if function.evaluate(x1) < function.evaluate(x2) else x2
    
    def getCentroid(x: np.array) -> np.array:
        # Calculate the centroid of all points but the last/worst one
        return np.mean(x[:-1], axis=0)
    
    def reflect(x_worst: np.array, x_centroid: np.array, alpha: float) -> np.array:
        # Reflect the worst point at the centroid by factor alpha
        return x_centroid + alpha * (x_centroid - x_worst)
    
    def expand(x_reflect: np.array, x_centroid: np.array, gamma: float) -> np.array:
        # Expand the reflected point by factor gamma
        return x_reflect + gamma * (x_reflect - x_centroid)
    
    def contract(x_min: np.array, x_centroid: np.array, beta: float) -> np.array:
        return x_min + beta * (x_centroid - x_min)
    
    def shrink(x: np.array, delta: float) -> np.array:
        # Shrink all points but the best one towards the best point by factor delta
        x_0 = x[0]
        return [xi+(x_0-xi)*delta for xi in x]
    
    def minimizeStep(self, x: np.array, function: IFunction, params: dict) -> np.array:
        # Calculate the next step of the minimization
        
        # 1. Calculate the centroid of all points but the last/worst one
        x_centroid = self.getCentroid(x)
        
        # 2. Reflect the worst point at the centroid
        x_worst = x[-1]
        x_reflect = self.reflect(x_worst, x_centroid, params['alpha'])
        
        # 3. If the reflected point is better than the best point, expand the reflected point
        if function.evaluate(x_reflect) < function.evaluate(x[0]):
            x_expand = self.expand(x_reflect, x_centroid, params['gamma'])
            x_expand_reflect_Min = self.evalMin(x_expand, x_reflect, function)
            return np.concatenate((x[:-1], [x_expand_reflect_Min]))
        
        # 4. If the reflected point is better than the second worst point, start new step with the reflected point
        if function.evaluate(x_reflect) < function.evaluate(x[-2]):
            return np.concatenate((x[:-1], [x_reflect]))
        
        # 5. Contract the minimum of the reflected point and the worst point
        x_worst_reflect_Min = self.evalMin(x_worst, x_reflect, function)
        x_contract = self.contract(x_worst_reflect_Min, x_centroid, params['beta'])
        
        # 6. If the contracted point is better than the worst point, return the contracted point
        if function.evaluate(x_contract) < function.evaluate(x[-1]):
            return np.concatenate((x[:-1], [x_contract]))
        
        # 7. Shrink
        return self.shrink(x, params['delta'])
    
    def minimize(self, function: IFunction, startingpoints: np.array, params: dict={'alpha':1.0, 'gamma':2.0, 'beta': 0.5, 'delta': 0.5}, iterations: int = 100, tol_x=1e-5, tol_y=1e-5) -> np.array:
        assert self.checkParams(params), "Assume all parameters are within bounds"
        x = startingpoints
        assert x.shape[0] == (function._domain.ambient_dimension + 1), "Assume number of starting points is equal to the dimension of the domain + 1"
        y = [function.evaluate(xi) for xi in x]
        assert y.shape[0] == 1, "Assume function is evaluating into a scalar"
        x = self.sortPoints(x, y)
        
        for i in range(iterations):
            # Schleifeninvarianz
            assert self.checkLinearIndependency(x), "Assume for all points x_i but x_0: x_i-x_0 linearly independent "
            x = self.minimizeStep(x, function, params)
            x = self.sortPoints(x, y)
            y = [function.evaluate(xi) for xi in x]
            if np.linalg.norm(x[0] - x[1]) < tol_x and np.linalg.norm(y(x[0]) - y(x[1])) < tol_y:
                break