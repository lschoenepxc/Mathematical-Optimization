# Mathematical Optimization

Dieses Repository enthält Implementierungen verschiedener mathematischer Optimierungsalgorithmen und -strukturen. Im Folgenden finden Sie eine Übersicht der Hauptklassen mit kurzen Beschreibungen und Beispielaufrufen.

## Klassenübersicht

### Allgemein
- [Sets](#1-sets)
- [(Differentiable) Functions](#2-differentiablefunction)
- [Scaled Differentiable Functions](#3-scaleddifferentiabelfunctions)

### Optimierungsalgorithmen
- [BFGS](#4-bfgs)
- [Bayesian Optimization](#5-bayesianoptimization)
- [Downhill Simplex](#6-downhillsimplex)
- [Feedforward NN](#7-feedforwardnn)
- [Gauß-Prozesse](#8-gp)
- [(Stochastic) Gradient Descent](#9-gradientdescent)
- [Linesearch](#10-linesearch)
- [Quadratic Programming](#11-qp)
- [Sequential Quadratic Programming](#12-sqp)

### SAT-Problem
- [Literale, Clauses, CNF](#13-literal)
- [DPLL](#14-dpll)

### 1. Sets
Beinhaltet verschiedene Mengenstrukturen, wie z.B. affine Räume und multidimensionale Intervalle, die in Optimierungsproblemen verwendet werden.

**Beispielaufruf aus `test_Sets.py`:**

```python
import numpy as np
from Set import AffineSpace, MultidimensionalInterval
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction


def test_affine_space():
    affine3 = AffineSpace(ambient_dimension=3)
    print(affine3._ambient_dimension == 3) # True
    print(affine3.contains(np.array([1, 2, 3]))) # True

def test_multidimensional():
    # multidimensional interval [lower_bounds[0], upper_bounds[0]] x [lower_bounds[1], upper_bounds[1]] x ...

    set = MultidimensionalInterval(lower_bounds=np.array([-1, -1]), upper_bounds=np.array([0, 42]))
    print(set.contains(np.array([-1, -1]))) # True
    print(set.contains(np.array([-1, 42]))) # True
    print(set.contains(np.array([[-1, 42]]))) # False

def test_bounded_set():
    # Bounded set in an affine space; given as intersection of a MultidimensionalInterval and an inequality constraint.
    # An inequality constraint is given by a differentiable functions f where x satisfies the constraint if all components of f(x) are non-positive.
    R = AffineSpace(1)
    f = DifferentiableFunction(
        name="x->x^2-1", domain=R, evaluate=lambda x: np.array([x[0]**2-1]), jacobian=lambda x: np.array([[2*x[0]]]))
    set = BoundedSet(lower_bounds=np.array([-2]), upper_bounds=np.array([2]), InequalityConstraints=f)

    print(set.contains(np.array([-1]))) # True
    print(set.contains(np.array([-1.0]))) # True
    print(set.contains(np.array([1]))) # True
    print(set.contains(np.array([0]))) # True
    print(set.contains(np.array([-1.5]))) # False
    print(set.contains(np.array([-2.0]))) # False
    print(set.contains(np.array([1.5]))) # False
    print(set.contains(np.array([2]))) # False

```

### 2. (Differentiable) Function
Stellt eine (differenzierbare) Funktion dar und bietet Methoden zur Berechnung von Wert (und Ableitung).

**Beispielaufruf:**

```python
from Function import Function
from DifferentiableFunction import DifferentiableFunction

def test_function():
    R = AffineSpace(1)
    f = Function(
        name="x->x^2", domain=R,
        evaluate=lambda x: np.array([x[0]**2]))
    ff = Function.FromComposition(f, f)
    # f.name = '(x->x^2) ° (x->x^2)'
    f2 = f+f
    # f.name = '(x->x^2) + (x->x^2)'
    f3 = 3*f
    # f.name = '3 * (x->x^2)'

    f.evaluate(np.array[0,1])

def test_differentiable_function():
    R = AffineSpace(1)
    f = DifferentiableFunction(
        name="x->x^2", domain=R,
        evaluate=lambda x: np.array([x[0]**2]),
        jacobian=lambda x: np.array([[2*x[0]]]))
    ff = DifferentiableFunction.FromComposition(f, f)
    # f.name = '(x->x^2) ° (x->x^2)'
    f2 = f+f
    # f.name = '(x->x^2) + (x->x^2)'
    f3 = 3*f
    # f.name = '3 * (x->x^2)'

    f.evaluate(np.array[0,1])

```

### 3. ScaledDifferentiableFunctions
Ermöglicht die Skalierung von differenzierbaren Funktionen, um deren Eingabe- und Ausgabebereiche anzupassen.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from ScaledDifferentiableFunction import ScaledDifferentiableFunction

def test_scaling(self):
    R2 = AffineSpace(2)
    f = DifferentiableFunction(
        name="f",
        domain=R2,
        evaluate=lambda x: np.array([x[0]**2, x[1]**2]),
        jacobian=lambda x: np.array([[2*x[0], 0], [0, 2*x[1]]])
    )
        
    # Skalierungsfaktoren und Offsets
        
    input_scalar = 2.0
    input_offset = 0
    output_scalar = np.array([3.0])
    output_offset = np.array([1.0, 1.0])
        
    sdf = ScaledDifferentiableFunction()
        
    # Funktion erstellen, die sowohl den Input als auch den Output skaliert und verschiebt
    scaled_offset_f = sdf.getScaledFunction(f, input_scalar=input_scalar, input_offset=input_offset, output_scalar=output_scalar, output_offset=output_offset)
```


### 4. BFGS
Implementiert den Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithmus, ein iteratives Verfahren zur Lösung von nichtlinearen Optimierungsproblemen ohne Nebenbedingungen.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from BFGS import BFGS

def test_bfgs():
    R = AffineSpace(1)
    f = DifferentiableFunction(
        name="x->x^2", domain=R,
        evaluate=lambda x: np.array([x[0]**2]),
        jacobian=lambda x: np.array([[2*x[0]]]))
    bfgs = BFGS()
    x = bfgs.Minimize(f, startingpoint=np.array([10.0]))
```

### 5. BayesianOptimization
Implementiert die Bayessche Optimierung, eine Strategie zur globalen Optimierung von Funktionen, die teuer zu evaluieren sind.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from BayesianOptimization import BO

def test_bayesian_optimization():
    R = AffineSpace(1)
    f = DifferentiableFunction(
        name="x->x^2", domain=R,
        evaluate=lambda x: np.array([x[0]**2]),
        jacobian=lambda x: np.array([[2*x[0]]]))
    bo = BO()
    x = bo.Minimize(f)
    print(x)
```

### 6. DownhillSimplex
Implementiert den Nelder-Mead-Algorithmus (Downhill Simplex), ein Verfahren zur Minimierung von Funktionen ohne Ableitungsinformationen.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from Function import Function
from DownhillSimplex import DownhillSimplex

def test_downhill_simplex():
    R = AffineSpace(1)
        function = Function(name="x->x^2", domain=R, evaluate=lambda x: np.array([x[0]**2]))
        startingpoints = np.array([[10.0], [3.0]])
        simplex = DownhillSimplex()
        params = {'alpha': 1.0, 'gamma': 2.0, 'beta': 0.5, 'delta': 0.5}
        result = simplex.minimize(function, startingpoints, params, iterations=100, tol_x=1e-5, tol_y=1e-5)
```

### 7. FeedForwardNN
Implementiert ein einfaches Feedforward-Neuronales Netzwerk.

**Beispielaufruf:**

```python
import numpy as np
from FeedForwardNN import ReLUFeedForwardNN
from GradientDescent import GradientDescent

def test_feedforward_nn():
    nn = ReLUFeedForwardNN()
    data_x = ...        # np.array
    data_y = ...        # np.array
    other_data = ...    # np.array

    gd = GradientDescent()
    params = nn.params
    # GD
    # loss = nn.ToLoss(data_x, data_y)
    # params = gd.Minimize(function=loss, startingpoint=params,
    #                      iterations=1000, learningrate=1e-2)

    # SGD    
    params = gd.StochasticMinimize(toLoss=nn.ToLoss, data_x=data_x, data_y=data_y, startingpoint=params, iterations=1000, learningrate=1e-2, batch_size=4)

    nn.params = params
    nn.ToFunction().evaluate(other_data)
```

### 8. GP
Implementiert Gaußsche Prozesse für nichtparametrische Regressionsaufgaben.

**Beispielaufruf aus `test_GP.py`:**

```python
import numpy as np
from GP import GP

def test_gp():
    data_x =        # np.array
    data_y =        # np.array
    # RBF Kernel
    gp = GP(data_x=data_x, data_y=data_y, kernel=GP.RBF())
    # oder 
    GP(data_x=data_x, data_y=data_y)
    # Matern Kernel
    gp = GP(data_x=data_x, data_y=data_y, kernel=GP.MaternCovariance())
    # Linear Kernel
    gp = GP(data_x=data_x, data_y=data_y, kernel=GP.Linear())

    mu = gp.PosteriorMean()
    sigma2 = gp.PosteriorVariance()
    sigma = gp.PosteriorStandardDeviation()
```

### 9. GradientDescent
Implementiert den Gradientenabstiegsalgorithmus zur Minimierung differenzierbarer Funktionen.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from DifferentiableFunction import DifferentiableFunction
from GradientDescent import GradientDescent

def test_gradient_descent():
    f = DifferentiableFunction(...)
    GD = GradientDescent()
    x = GD.Minimize(f, startingpoint=np.array([10.0]), learningrate=1.0)
```

### 10. LineSearch
Bietet Methoden zur Durchführung von LineSearch in Optimierungsalgorithmen. Gibt eine Schrittweite alpha zurück.

**Beispielaufruf:**

```python
import numpy as np
from DifferentiableFunction import DifferentiableFunction
from Set import AffineSpace, MultidimensionalInterval
import LineSearch

def test_line_search():
    linesearch = LineSearch.LineSearch()
    alpha = linesearch.LineSearchForWolfeConditions(
            function, startingpoint=x, direction=p,
            lower_bounds=lower_bounds, upper_bounds=upper_bounds)
```

### 11. QP
Implementiert Verfahren zur Lösung von Quadratischen Programmierungsproblemen (QP).

**Beispielaufruf aus `SQP.py`:**

```python
import numpy as np
from QP import QP
from Set import AffineSpace, MultidimensionalInterval
from DifferentiableFunction import DifferentiableFunction

def test_qp():
    qp = QP()
    x = startingpoint
    gradient = function.jacobian(x).reshape([-1])
    H = np.identity(n)  # Approximate Hessian

    # linearize problem and solve QP for search direction
    ineq_eval = ineq.evaluate(x)
    ineq_jacobian = ineq.jacobian(x)
    p = qp.QP(H, gradient.transpose(), ineq_jacobian, -ineq_eval)
```

### 12. SQP
Implementiert das Sequential Quadratic Programming (SQP), ein Verfahren zur Lösung nichtlinearer Optimierungsprobleme mit Nebenbedingungen.

**Beispielaufruf:**

```python
import numpy as np
from Set import AffineSpace
from SetsFromFunctions import BoundedSet
from DifferentiableFunction import DifferentiableFunction
from SQP import SQP

def test_SQP():
    sqp = SQP()

    R = AffineSpace(2)
    X = DifferentiableFunction(
        name="x", domain=R, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
    Y = DifferentiableFunction(
        name="y", domain=R, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))
    const = X**2+Y**2-3
    domain = BoundedSet(lower_bounds=np.array(
        [-2, -2]), upper_bounds=np.array([2, 2]), InequalityConstraints=const)
    X = DifferentiableFunction(
        name="x", domain=domain, evaluate=lambda x: np.array([x[0]]), jacobian=lambda x: np.array([[1, 0]]))
    Y = DifferentiableFunction(
        name="y", domain=domain, evaluate=lambda x: np.array([x[1]]), jacobian=lambda x: np.array([[0, 1]]))

    f = (X-10)**2+(Y-10)**2
    x = sqp.Minimize(f, startingpoint=np.array([0.1, 1.0]))
```

### 13. Literal
Repräsentiert ein Literal, Clauses und CNFs in der Aussagenlogik, verwendet im DPLL-Algorithmus.

**Beispielaufruf:**

```python
from Literal import Literal, Clause, CNF

def test_literal():
    a = Literal("a")
    b = Literal("b")
    not_a = -a
    not_b = -b

    clauses = {
        Clause({a, b}),
        Clause({not_a, b}),
        Clause({a, not_b})
    }
    cnf = CNF(clauses)
```

### 14. DPLL
Implementiert den Davis-Putnam-Logemann-Loveland (DPLL) Algorithmus, ein Backtracking-Verfahren zur Erfüllbarkeitsprüfung in der Aussagenlogik.

**Beispielaufruf:**

```python
from Literal import Literal, Clause, CNF
from DPLL import DPLL

def test_dpll():
    a = Literal("a")
    b = Literal("b")
    not_a = -a
    not_b = -b

    clauses = {
        Clause({a, b}),
        Clause({not_a, b}),
        Clause({a, not_b})
    }

    cnf = CNF(clauses)
    dpll = DPLL()
    satisfiable, assignment = dpll.dpll(cnf, {})
```