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

BFGS (Broyden-Fletcher-Goldfarb-Shanno)

- Vorteile: eignet sich (gut) für hochdimensionale Probleme, da zweite Ableitung nicht explizit berechnet wird.

- Nachteile: kann speicherintensiv werden, da es eine *(n x n)* Hesse-Matrix speichert.

Bayesian Optimization

- Vorteile: nützlich für teure Zielfunktionen, da Anzahl der Funktionsauswertungen minimiert werden; verwendet probabilistische Modelle (hier: GP), um Zielfunktion zu approximieren.
- Nachteile: Skalierbarkeit von Bayesian Optimization begrenzt, da Komplexität der GPs mit der Anzahl der Dimensionen und Datenpunkten zunimmt.

Stochastic Gradient Descent (SGD)

- Vorteile: sehr effizient für hochdimensionale Probleme (große Datensätze); verwendet stochastische Approximationen des Gradienten (Reduzierung der Berechnungskosten pro Iteration)
- Nachteile: kann bei schlecht konditionierten Problemen langsam konvergieren; erfordert sorgfältige Wahl der Hyperparameter 

Sequential Quadratic Programming (SQP)

- Vorteile: löst Reihe von quadratischen Unterproblemen, die einfacher zu handhaben sind.
- Nachteile: sehr hohen Dimensionen --> speicherintensiv und rechenaufwendig

Downhill Simplex (Nelder-Mead)

- Nachteile: skaliert schlecht mit der Dimension des Problems, da die Anzahl der Simplex-Ecken mit der Dimension zunimmt --> exponentielle Zunahme der Berechnungskosten

#### Mögliche Anpassungen

- **BoundedSets (in BO)**: kann Suchraum beschränken; kann Effizienz erhöhen, indem unnötige Bereiche ausgeschlossen werden
- **Sampling**: repräsentative Stichprobe des Suchraums --> kann die Effizienz erhöhen durch Reduzierng der Berechnungskosten, indem nur Teilmenge der möglichen Punkte untersucht wird
- **Principal Component Analysis**: Reduzierung der Dimension des Suchraums, indem wichtigste Komponenten extrahiert werden

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



