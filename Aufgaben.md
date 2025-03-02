# Aufgaben

**Laura Schöne, Matr.-Nr.: 15480020**

## Aufgabe 1 (+ Bonusaufgabe): Downhill Simplex

Downhill Simplex siehe: 
- `DownhillSimplex.py`, 
- `test_DHSimplex.py`, 
- `project`-Methode in `SetsFromFunctions.py`, 
- Trennung von `Functions` und `DifferentiableFunctions` in `Functions.py`

## Aufgabe 2.2: Skalierungen von Funktionen

Skalierung:
-  weil: Fließkommazahlen arbeiten am besten im Bereich [−1,1]
- mögliche Probleme bei: 
    - BFGS (approximierte Hesse-Matrix), 
    - Gauß-Prozesse und BO (Kernel-Funktionen),
    - NN (langsamere Konvergenz, schlechtere Modellanpassungen)
    - (S)GD (numerische Instabilitäten bei Gradienteninformationen)
- hier: Funktion wird so sklaiert, dass die Ausgabewerte in [-1,1] liegen und so verschoben, dass sie Eingaben aus [-1,1] erhält
- Einschränkung für Auto-Scaling: Funktion sollte auf einem MultidimensionlInterval definiert sein
- Max und Min Werte der Funktion für die Output-Skalierung entweder durch Sampling oder BO

Siehe: `ScaledDifferentiableFunctions.py`, Tests in `test_Functions.py`

## Aufgabe 3: SGD mit NNS

SGD siehe:
- `GradientDescent.py`, Methode `StochasticMinimize`
- Test mit NN in `test_NN.py`
- Complexity `ToLoss` in `FeedForwardNN.py`

## Aufgabe 4: Funktionsklassen und Funktionsmultiplikation

Funktionsklassen: sin, cos, tan, exp, log, sqrt, sigmoid, square, cube, arccos, own_function siehe:
- `DifferentiableFunction.py`
- `__mul__`-Methode für 2 Funktionen, die eindimensional sind in `Function.py`, `DifferentiableFunction.py`
- Test in `test_Functions.py`

## Aufgabe 5: Interfaces für Körper, Vektorräume und Matrizen (als lineare Abbildungen) nach Curry-Howard

Fields, VectorSpaces, LinearMap, Gauss siehe
- `Field.py`
- `test_Fields.py`

## Aufgabe 6: Profiling BFGS

siehe
- `profiling_BFGS.py`
- `profile_BFGS.png`

## Aufgabe 7: Hochdimensionale Daten in den Optimierungsalgorithmen

#### Geeignete Optimierungsverfahren

BFGS (Broyden-Fletcher-Goldfarb-Shanno)

- Vorteile: BFGS ist ein Quasi-Newton-Verfahren, das die Hesse-Matrix approximiert und somit effizienter als Newton-Verfahren ist. Es eignet sich gut für hochdimensionale Probleme, da es die zweite Ableitung nicht explizit berechnet.

- Nachteile: BFGS kann bei sehr hohen Dimensionen speicherintensiv werden, da es eine ( n \times n ) Hesse-Matrix speichert.

Bayesian Optimization

- Vorteile: Bayesian Optimization ist besonders nützlich für teure Zielfunktionen, da es die Anzahl der Funktionsauswertungen minimiert. Es verwendet probabilistische Modelle (hier: Gaussian Processes), um die Zielfunktion zu approximieren.
- Nachteile: Die Skalierbarkeit von Bayesian Optimization ist begrenzt, da die Komplexität der Gaussian Processes mit der Anzahl der Dimensionen und Datenpunkten zunimmt.

Stochastic Gradient Descent (SGD)

- Vorteile: SGD ist sehr effizient für hochdimensionale Probleme, insbesondere bei großen Datensätzen. Es verwendet stochastische Approximationen des Gradienten, was die Berechnungskosten pro Iteration reduziert.
- Nachteile: SGD kann bei schlecht konditionierten Problemen langsam konvergieren und erfordert sorgfältige Wahl der Hyperparameter (z.B. Lernrate).

Sequential Quadratic Programming (SQP)

- Vorteile: SQP ist ein leistungsfähiges Verfahren für nichtlineare Optimierungsprobleme mit Nebenbedingungen. Es löst eine Reihe von quadratischen Unterproblemen, die einfacher zu handhaben sind.
- Nachteile: SQP kann bei sehr hohen Dimensionen speicherintensiv und rechenaufwendig sein.

#### Weniger geeignete Optimierungsverfahren

Downhill Simplex (Nelder-Mead)

- Nachteile: Der Downhill Simplex Algorithmus skaliert schlecht mit der Dimension des Problems, da die Anzahl der Simplex-Ecken mit der Dimension zunimmt. Dies führt zu einer exponentiellen Zunahme der Berechnungskosten und kann die Konvergenz verlangsamen.

Line Search

- Nachteile: Line Search Verfahren sind in der Regel für unidimensionale Optimierungsprobleme konzipiert und erfordern eine geeignete Suchrichtung. In hochdimensionalen Räumen kann die Wahl der Suchrichtung und die Durchführung der Line Search ineffizient sein.

#### Empfehlungen
Für hochdimensionale zulässige Mengen empfehlen sich insbesondere Verfahren wie BFGS, Bayesian Optimization und Stochastic Gradient Descent. Diese Verfahren sind effizienter und skalierbarer als andere Methoden. Es ist jedoch wichtig, die spezifischen Anforderungen und Eigenschaften des Optimierungsproblems zu berücksichtigen, um das am besten geeignete Verfahren auszuwählen.

#### Mögliche Anpassungen
????

BoundedSets in BO? Skalierte Funktionen in BO? Sampling?

## Aufgabe 8: Cholesky in GP

siehe: 
- `GP.py`

## Aufgabe 9: GP Kerne

siehe:
- RBF-Kernel, MaternKernel, LinearKernel in `GP.py`
- Tests in `test_GP.py`

Eine Ableitung wird nicht benötigt, daher werden die Kernel-Funktionen als lamdas definiert.
Ergebnisse `test_GP_kernel_comparison_linear_data` und `test_GP_kernel_comparison_noisy_exponential_data`:
- RBF und Matern Kerne: konsistente und zuverlässige Vorhersagen sowohl für lineare als auch für verrauschte exponentielle Daten. Sie scheinen gut mit nichtlinearen Daten umzugehen.
- Linear Kernel: Dieser Kernel funktioniert gut bei linearen Daten, jedoch schlechte Performance bei verrauschten exponentiellen Daten, was zu stark abweichenden Vorhersagen führt.

## Aufgabe 10: BO Schnittstelle für vorhandene Daten

siehe: `BayesianOptimization.py`

## Aufgabe 11: Varianten von TSP

siehe `aufgabe_TSP.md`

## Bonusaufgabe: DPLL

- `Literal`, `Clause`, `CNF` als Datenstrukturen in `Literal.py`
- `DPLL.py` für den Algorithmus
- Tests in `test_Literals.py`, `test_DPLL.py`

Vergleichbarkeit zu anderen Implementierungen: `test_DPLL_PySAT.py`
Nicht gegeben. Die Handhabung über die Klassen `Literals`, `Clause` und `CNF` erzeugen einen zu großen Overhead, daher ist die Laufzeit bedeutend höher für die hier selbst implementierte Lösung.



