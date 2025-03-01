# Mathematical Optimization

Exercises for the course Mathematical Optimization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Bayesian Optimization](#bayesian-optimization)
- [BFGS](#bfgs)
- [Differentiable Functions](#differentiable-functions)

## Installation

Instructions on how to install the dependencies for this project.

```
# Example command to install dependencies
pip install -r requirements.txt
```

## Usage

Instructions on how to use the project.

```
# Example command to run all the tests for this project
python pytest
```

## Bayesian Optimization
Bayesian Optimization (BO) ist eine Methode zur globalen Optimierung von Funktionen, die teuer zu bewerten sind. BO verwendet ein probabilistisches Modell, typischerweise einen Gaussian Process (GP), um eine Surrogatfunktion der Zielfunktion zu erstellen. Diese Surrogatfunktion wird verwendet, um eine Akquisitionsfunktion zu definieren, die die nächste zu bewertende Stelle vorschlägt. Das Ziel ist es, die Zielfunktion mit möglichst wenigen Bewertungen zu minimieren.

1. **Initialisierung**:

   Die Methode `Minimize` der Klasse `BO` nimmt eine differenzierbare Funktion (`IDifferentiableFunction`), die Anzahl der Iterationen und optionale Anfangsdaten (`x` und `y`) als Eingaben.
   Der Definitionsbereich der Funktion wird aus der Funktion extrahiert.
   Falls Anfangsdaten vorhanden sind, werden diese überprüft und initialisiert. Andernfalls werden leere Arrays für `data_x` und `data_y` erstellt.

2. **Gaussian Process (GP)**:

   Ein GP-Modell wird mit den initialen Daten (`data_x` und `data_y`) erstellt. Der GP dient als Surrogatmodell der Zielfunktion.

3. **Sequential Quadratic Programming (SQP)**:

   Ein SQP-Optimierer wird erstellt, um die Akquisitionsfunktion zu minimieren.

4. **Iterative Optimierung**:

   Für eine gegebene Anzahl von Iterationen wird die Akquisitionsfunktion definiert, die den Posterior Mean und die Posterior Standard Deviation des GP-Modells berücksichtigt.
   Der SQP-Optimierer wird verwendet, um die Akquisitionsfunktion zu minimieren und den nächsten Punkt (`x`) zu bestimmen, an dem die Zielfunktion bewertet werden soll.
   Die Zielfunktion wird an diesem Punkt bewertet, und die neuen Daten (`x` und `y`) werden zu den vorhandenen Daten hinzugefügt.
   Das GP-Modell wird mit den aktualisierten Daten neu trainiert.

5. **Ergebnis**:

   Nach den Iterationen wird der Punkt zurückgegeben, an dem die Zielfunktion den minimalen Wert erreicht hat.

### Dependencies in diesem Projekt

1. **Gaussian Processes (GPs)**:
   GPs werden verwendet, um ein probabilistisches Modell der Zielfunktion zu erstellen. Sie bieten Vorhersagen über den Mittelwert und die Unsicherheit der Funktion an verschiedenen Punkten im Definitionsbereich.

2. **Sequential Quadratic Programming (SQP)**:
   SQP wird verwendet, um die Akquisitionsfunktion zu minimieren. Es handelt sich um ein Optimierungsverfahren, das für nichtlineare Optimierungsprobleme mit Beschränkungen geeignet ist.

3. **Differentiable Functions**:
   Die Zielfunktion, die minimiert werden soll, muss eine differenzierbare Funktion sein (`IDifferentiableFunction`). Dies ermöglicht die Verwendung von Gradienteninformationen in der Optimierung.

## BFGS
BFGS (Broyden-Fletcher-Goldfarb-Shanno) ist ein Quasi-Newton-Verfahren zur numerischen Lösung von nichtlinearen Optimierungsproblemen. Es gehört zu den iterativen Verfahren, die die Hesse-Matrix der Zielfunktion approximieren, um die Richtung und Größe der Schritte zu bestimmen. BFGS ist besonders nützlich für Probleme, bei denen die Berechnung der exakten Hesse-Matrix zu teuer oder unmöglich ist.

1. **Initialisierung**:

Die Methode `Minimize` der Klasse `BFGS` nimmt eine differenzierbare Funktion (`IDifferentiableFunction`), einen Startpunkt (`startingpoint`), die Anzahl der Iterationen und Toleranzwerte (`tol_x` und `tol_y`) als Eingaben.
Der Startpunkt `x` wird initialisiert, und die Dimension `n` wird bestimmt.
Die Hesse-Matrix `H` wird als Einheitsmatrix initialisiert.
Der Funktionswert `y` und der Gradient `gradient` am Startpunkt werden berechnet.
Falls die Funktion auf einem `MultidimensionalInterval` definiert ist, werden die unteren und oberen Schranken (`lower_bounds` und `upper_bounds`) extrahiert. Andernfalls werden sie auf `-∞` und `+∞` gesetzt.

2. **Iterative Optimierung**:

Für eine gegebene Anzahl von Iterationen wird der Gradient überprüft. Wenn der Gradient null ist, wird der aktuelle Punkt `x` zurückgegeben.
Die Suchrichtung `p` wird als negatives Produkt der Hesse-Matrix `H` und des Gradienten `gradient` berechnet.
Eine Line Search wird durchgeführt, um die Schrittweite `alpha` zu bestimmen, die die Wolfe-Bedingungen erfüllt.
Der neue Punkt `x` wird durch Hinzufügen des Schritts `s = alpha * p` zum aktuellen Punkt berechnet und auf die zulässigen Schranken projiziert.
Wenn die Norm des Schritts `s` kleiner als `tol_x` ist, wird die Iteration abgebrochen.
Der Funktionswert `y_new` am neuen Punkt wird berechnet, und die Differenz `delta_y` zwischen dem alten und neuen Funktionswert wird überprüft. Wenn `delta_y` kleiner als `tol_y` ist, wird die Iteration abgebrochen.
Der alte Gradient `gradient_old` wird gespeichert, und der neue Gradient `gradient` wird berechnet.
Die Differenz `delta_grad` zwischen dem neuen und alten Gradienten wird berechnet.
Die Skalierung `scaling` wird als Skalarprodukt von `s` und `delta_grad` berechnet. Wenn `scaling` positiv ist, wird die Hesse-Matrix `H` mit der BFGS-Formel aktualisiert.

3. **Ergebnis**:

Nach den Iterationen wird der Punkt `x` zurückgegeben, an dem die Zielfunktion den minimalen Wert erreicht hat.

### Dependencies in diesem Projekt

1. **Differentiable Functions**:
Die Zielfunktion, die minimiert werden soll, muss eine differenzierbare Funktion sein (`IDifferentiableFunction`). Dies ermöglicht die Verwendung von Gradienteninformationen in der Optimierung.

2. **Line Search**:
Eine Line Search wird verwendet, um die Schrittweite zu bestimmen, die die Wolfe-Bedingungen erfüllt. Dies verbessert die Konvergenz des BFGS-Verfahrens.

3. **Sets (MultidimensionalInterval)**:
Falls die Funktion auf einem `MultidimensionalInterval` definiert ist, werden die unteren und oberen Schranken (`lower_bounds` und `upper_bounds`) verwendet, um die Punkte auf die zulässigen Schranken zu projizieren.

## Differentiable Functions
Die Klasse `IDifferentiableFunction` und ihre Implementierung `DifferentiableFunction` modellieren differenzierbare Funktionen von einer Menge (`ISet`) nach `R^n`. Diese Klassen bieten eine einheitliche Schnittstelle zur Definition und Manipulation von Funktionen und deren Ableitungen. Dies ist besonders nützlich für Optimierungsverfahren, die Gradienteninformationen benötigen.

1. **Definition**:

Die Klasse `IDifferentiableFunction` erbt von `IFunction` und fügt eine abstrakte Methode `jacobian` hinzu, die die Jacobian-Matrix der Funktion an einem gegebenen Punkt berechnet.
Die Klasse `DifferentiableFunction` implementiert `IDifferentiableFunction` und ermöglicht die Konstruktion von differenzierbaren Funktionen aus Lambdas für die Funktion selbst und deren Ableitungen.

2. **Operationen**:

- **Addition**: Die Klassen unterstützen die Addition von Funktionen und Konstanten. Die Jacobian-Matrix der resultierenden Funktion wird entsprechend angepasst.
Multiplikation: Die Klassen unterstützen die Multiplikation von Funktionen und Konstanten. Die Jacobian-Matrix der resultierenden Funktion wird entsprechend angepasst.
- **Potenzierung**: Die Klassen unterstützen die Potenzierung von Funktionen mit ganzzahligen Exponenten. Die Jacobian-Matrix der resultierenden Funktion wird entsprechend angepasst.
- **Subtraktion**: Die Klassen unterstützen die Subtraktion von Funktionen. Dies wird durch die Addition der negativen Funktion erreicht.
- **Paarung**: Die Klassen unterstützen die Paarung von Funktionen, wobei die Jacobian-Matrizen der beiden Funktionen zusammengeführt werden.
- **Kartesisches Produkt**: Die Klassen unterstützen das kartesische Produkt von Funktionen, wobei die Jacobian-Matrizen der beiden Funktionen zusammengeführt werden.
- **Multipliaktion**: Die Klassen unterstützen die Multiplikation von Funktionen mit Skalaren und die Multiplkation von Funktionen, die im eindimensionalen definiert sind.

3. **Spezielle Funktionen**:

Die Klassen bieten Methoden zur Konstruktion spezieller Funktionen wie Projektionen, Identitätsfunktionen, ReLU, und verschiedene elementare Funktionen (sin, cos, tan, exp, log, sqrt, sigmoid, square, cube, arccos).
Eine benutzerdefinierte Funktion `own_function` wird ebenfalls bereitgestellt, die eine komplexe Kombination mehrerer elementarer Funktionen darstellt.

### Dependencies in diesem Projekt

1. **Sets (ISet, AffineSpace)**:
Die Klassen `ISet` und `AffineSpace` definieren die Mengen, auf denen die differenzierbaren Funktionen definiert sind. Diese Mengen bieten Methoden zur Überprüfung der Zulässigkeit von Punkten und zur Projektion von Punkten auf die Menge.

2. **Function**:
Die Klasse `Function` bietet eine Basisklasse für allgemeine Funktionen und stellt Methoden zur Verfügung, die in `IDifferentiableFunction` und `DifferentiableFunction` erweitert werden.

3. **Multimethod**:
Die Bibliothek `multimethod` wird verwendet, um Methodenüberladung für die Operatoren `__add__`, `__mul__` und andere zu ermöglichen. Dies erleichtert die Definition von Operationen zwischen differenzierbaren Funktionen und Konstanten.

## Downhill Simplex
Der Downhill Simplex Algorithmus, auch als Nelder-Mead-Verfahren bekannt, ist ein nichtlineares Optimierungsverfahren, das keine Ableitungen benötigt. Es ist besonders nützlich für Optimierungsprobleme, bei denen die Zielfunktion nicht differenzierbar oder schwer zu differenzieren ist. Der Algorithmus verwendet ein Simplex, eine geometrische Figur mit \(n+1\) Ecken in einem \(n\)-dimensionalen Raum, um die Zielfunktion zu minimieren.

1. **Initialisierung**:

   Die Methode `minimize` der Klasse `DownhillSimplex` nimmt eine Funktion (`IFunction`), Startpunkte (`startingpoints`), Parameter (`params`), die Anzahl der Iterationen und Toleranzwerte (`tol_x` und `tol_y`) als Eingaben.
   Die Startpunkte `x` werden initialisiert und die Dimension des Simplex wird überprüft.
   Die Funktionswerte `y` an den Startpunkten werden berechnet und die Punkte werden nach ihren Funktionswerten sortiert.

2. **Iterative Optimierung**:

   Für eine gegebene Anzahl von Iterationen wird die Linearunabhängigkeit der Punkte überprüft.
   Der Algorithmus führt verschiedene Schritte durch, um das Simplex zu aktualisieren:
     - **Reflexion**: Der schlechteste Punkt wird am Schwerpunkt der anderen Punkte reflektiert.
     - **Expansion**: Wenn der reflektierte Punkt besser als der beste Punkt ist, wird der reflektierte Punkt weiter in die gleiche Richtung expandiert.
     - **Kontraktion**: Wenn der reflektierte Punkt nicht besser als der zweitbeste Punkt ist, wird der schlechteste Punkt in Richtung des Schwerpunkts kontrahiert.
     - **Schrumpfung**: Wenn die Kontraktion nicht zu einer Verbesserung führt, werden alle Punkte in Richtung des besten Punktes geschrumpft.
   Die Punkte werden nach jedem Schritt neu sortiert und die Funktionswerte werden aktualisiert.
   Wenn die Norm der Differenz zwischen den besten und zweitbesten Punkten kleiner als `tol_x` ist und die Norm der Differenz der Funktionswerte kleiner als `tol_y` ist, wird die Iteration abgebrochen.

3. **Ergebnis**:

   Nach den Iterationen wird der Punkt `x` zurückgegeben, an dem die Zielfunktion den minimalen Wert erreicht hat.

### Dependencies in diesem Projekt

1. **Function**:
   - Die Zielfunktion, die minimiert werden soll, muss eine Funktion sein (`IFunction`). Dies ermöglicht die Bewertung der Funktion an verschiedenen Punkten im Simplex.

2. **Sets (AffineSpace, MultidimensionalInterval)**:
   - Die Klassen `AffineSpace` und `MultidimensionalInterval` definieren die Mengen, auf denen die Funktion definiert ist. Diese Mengen bieten Methoden zur Überprüfung der Zulässigkeit von Punkten und zur Projektion von Punkten auf die Menge.

## DPLL
Der DPLL-Algorithmus (Davis-Putnam-Logemann-Loveland) ist ein Backtracking-Algorithmus zur Bestimmung der Erfüllbarkeit von Aussagenlogik-Formeln in konjunktiver Normalform (CNF). Er erweitert den ursprünglichen Davis-Putnam-Algorithmus durch die Einführung von Unit Propagation und Pure Literal Elimination, um die Effizienz zu verbessern.

1. **Unit Propagation**:

   Die Methode `unit_propagate` vereinfacht die CNF-Formel, indem sie Einheitsklauseln (Klauseln mit nur einem Literal) identifiziert und die entsprechenden Variablen zuweist.
   Die Methode entfernt Klauseln, die das zugewiesene Literal enthalten, und entfernt das negierte Literal aus den verbleibenden Klauseln.

2. **Pure Literal Elimination**:

   Die Methode `pure_literal_assign` identifiziert pure Literale (Literale, die nur in einer Polarität vorkommen) und weist ihnen Werte zu.
   Die Methode entfernt Klauseln, die das pure Literal enthalten.

3. **DPLL-Algorithmus**:

   Die Methode `dpll` implementiert den eigentlichen DPLL-Algorithmus, um zu bestimmen, ob eine CNF-Formel erfüllbar ist.
   Der Algorithmus führt Unit Propagation und Pure Literal Elimination durch, um die Formel zu vereinfachen.
   Der Algorithmus überprüft die Stoppbedingungen:
     - Wenn die CNF-Formel leer ist, ist die Formel erfüllbar.
     - Wenn die CNF-Formel eine leere Klausel enthält, ist die Formel unerfüllbar.
   Der Algorithmus wählt ein Literal aus, das noch nicht zugewiesen wurde, und versucht, die Formel zu erfüllen, indem er das Literal auf `True` und `False` setzt.
   Wenn keine der Zuweisungen funktioniert, wird zurückverfolgt und das Literal wird entfernt.

4. **Ergebnis**:

   Die Methode `dpll` gibt ein Tupel zurück, das angibt, ob die Formel erfüllbar ist, und die entsprechende Variablenbelegung.

### Dependencies in diesem Projekt

1. **Literal, Clause, CNF**:
   Die Klassen `Literal`, `Clause` und `CNF` modellieren die Struktur der Aussagenlogik-Formeln in konjunktiver Normalform. Sie bieten Methoden zur Manipulation und Bewertung der Formeln.

## Feedforward Neural Network
Ein Feedforward Neural Network (FNN) ist ein künstliches neuronales Netzwerk, bei dem die Informationen nur in eine Richtung fließen – von den Eingabeknoten über die versteckten Knoten zu den Ausgabeknoten. Das ReLUFeedForwardNN verwendet die Rectified Linear Unit (ReLU) als Aktivierungsfunktion, um nichtlineare Transformationen zu ermöglichen.

1. **Initialisierung**:

   Die Klasse `ReLUFeedForwardNN` wird mit einer Methode `__init__` initialisiert, die die Dimensionen des Netzwerks (`dims`), die Eingabedimension (`input_dim`), die Dimension der versteckten Schicht (`hidden_dim`) und die ReLU-Aktivierungsfunktion definiert.
   Die Parameter des Netzwerks (`lins` und `bias`) werden zufällig initialisiert und als Vektoren gespeichert. Diese Parameter werden später in Matrizen umgeformt.

2. **Netzwerkfunktion**:

   Die Methode `ToFunction` gibt die differenzierbare Funktion zurück, die das Netzwerk von den Eingaben auswertet, nachdem die Parameter festgelegt wurden.
   Die Parameter werden in Matrizen umgeformt, und die linearen Transformationen (`f0` und `f1`) sowie die Bias-Translation (`bias`) werden definiert.
   Die Methode verwendet die Komposition von Funktionen, um die endgültige Netzwerkfunktion zu erstellen.

3. **Verlustfunktion**:

   Die Methode `ToLoss` gibt die differenzierbare Funktion zurück, die den Verlust des Netzwerks auf festen gegebenen Daten auswertet.
   Die Methode überprüft, ob die Anzahl der Datenpunkte mit der Anzahl der Labels übereinstimmt.
   Die Methode verwendet eine Schleife über die Datenpunkte, um die Verlustfunktion zu berechnen. Dies ist nicht für Batch-Training oder andere interessante Techniken optimiert.
   Die Methode verwendet das Kronecker-Produkt, um die lineare Transformation zu definieren, was in Bezug auf Speicher- und Zeitkomplexität problematisch sein kann.
   Die Methode berechnet den lokalen Verlust für jeden Datenpunkt und summiert ihn, um den Gesamtverlust zu erhalten.
   Der Gesamtverlust wird durch die Anzahl der Datenpunkte geteilt, um den mittleren quadratischen Fehler zu berechnen.

4. **Ergebnis**:

   Die Methode `ToLoss` gibt die Verlustfunktion zurück, die den mittleren quadratischen Fehler des Netzwerks auf den gegebenen Daten auswertet.

### Dependencies in diesem Projekt

1. **Differentiable Functions**:
Die Klasse `DifferentiableFunction` wird verwendet, um die linearen Transformationen, Bias-Translationen und ReLU-Aktivierungsfunktionen zu definieren. Diese Funktionen sind differenzierbar und ermöglichen die Verwendung von Gradienteninformationen in der Optimierung.

2. **Sets (AffineSpace)**:
Die Klasse `Set` definiert die Mengen, auf denen die Funktionen definiert sind. Diese Mengen bieten Methoden zur Überprüfung der Zulässigkeit von Punkten und zur Projektion von Punkten auf die Menge.
3. **Gradient Descent**:
Die Klasse `GradientDescent` wird verwendet, um die Parameter des Netzwerks zu optimieren, indem der Verlust minimiert wird.

