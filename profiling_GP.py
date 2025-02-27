import numpy as np
import scipy as sp
from GP import GP
import time


def simple_comparison():

    runs = 1000   

    passedTime = [0,0,0]

    for i in range(runs):
        

        # chat gpt proposed the following for a random positiv definit matrix:
        A = np.random.rand(100, 100)
        A_sym = np.dot(A, A.T)
        A_sym += np.eye(A_sym.shape[0])
        x = A_sym

        # y can be truly random
        y = np.random.rand(100)
        L = np.linalg.cholesky(x)
        
        # yeah, this is good code
        t0 = time.time()
        sp.linalg.cho_solve((L, True), y, check_finite=False)
        t1 = time.time()
        passedTime[0] = passedTime[0] + (t1 - t0)

        t0 = time.time()
        np.linalg.solve(L.T, np.linalg.solve(L, y))
        t1 = time.time()
        passedTime[1] = passedTime[1] + (t1 - t0)

        t0 = time.time()
        np.linalg.solve(x, y)
        t1 = time.time()
        passedTime[2] = passedTime[2] + (t1 - t0)
        
    print("time cho_solve_Ly:",passedTime[0],"s")
    print("time nnp.linalg.solve L,L.T, y:",passedTime[1],"s")
    print("time np.linalg.solve x, y:",passedTime[2],"s")    

def result_comparison():
    A = np.random.rand(3, 3)
    A_sym = np.dot(A, A.T)
    A_sym += np.eye(A_sym.shape[0])
    x = A_sym
    y = np.random.rand(3)
    L = np.linalg.cholesky(x)
    
    res1 = sp.linalg.cho_solve((L, True), y, check_finite=False)
    res2 = sp.linalg.cho_solve((x, True), y, check_finite=False)
    res3 = np.linalg.solve(L.T, np.linalg.solve(L, y))
    res4 = np.linalg.solve(x, y)
    print("result cho_solve L, y\n", res1)
    print("result cho_solve x, y\n", res2)
    print("result np.linalg.solve L,L.T, y\n", res3)
    print("result np.linalg.solve x, y\n", res4)
    


if __name__ == '__main__':
    result_comparison()
    # -> the results are not the same, so cho_solve really is specific to L and the correct solution to the problem
    simple_comparison()
    # -> scipy.linalg.cho_solve() is faster than np.linalg.solve(), at least for already decomposed matrices
    # This is exaxtly what we want, as replacing np.linalg.solve() with scipy.linalg.cho_solve() will boost performance
    