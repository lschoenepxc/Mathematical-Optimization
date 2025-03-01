# Varianten von TSP
Seit das himmlischen Großrechnersystem in die Wolke verlegt wurde, gibt es andauernd Abstürze.
Heute ist es mal wieder so weit, dabei wartet der Weihnachtsmann gerade jetzt ganz dringend auf den Ausdruck seiner Rundtour für die Weihnachtsnacht. 
Würde der Himmelsrechner funktionieren, wäre das alles gar kein Problem, schließlich wurde der Rechner vor gar nicht allzu langer Zeit von Alan entworfen und verfügt über diesen tollen Nichtdeterminismus. 
Nun aber muss der Weihnachtsmann ein Verfahren entwerfen, mit dem er auch nach irdischen Maßstäben eﬀizient seine Weihnachtstour berechnen kann.
Die Distanzen zwischen den einzelnen Orten, zu denen er muss, kennt er natürlich. Aber wie jedes Jahr stellt sich das Problem, dass er an keinem Ort zweimal auftauchen darf (wegen neugieriger Kinder). 
Außerdem würden seine Rentiere dauerhaft in Streik treten, wenn sie herausbekommen würden, dass sie nicht die kürzeste Route genommen haben
Glücklicherweise ist sein treues Leitrentier Rudolph in der Lage, zu jeder vorgelegten Eingabe von Orten und ihren Distanzen, eﬀizient zu entscheiden, ob es eine Tour mit den geforderten Eigenschaften der Länge höchstens b Kilometer gibt oder nicht. 
Allerdings wissen Rudolph und der Weihnachtsmann noch nicht, wie sie daraus eine Lösung für das ursprüngliche Problem entwickeln können.

**Formal: Zeige, falls die Entscheidungsvariante des TSP in P ist, so ist auch das TSP in P.**

Beweis:

Leitrentier Rudolph kann in polynomieller Zeit entscheiden, 
ob es eine Tour zu jeder vorgelegten Eingabe von Orten (Anzahl `n`) und ihren Distanzen mit der Länge höchstens `b` Kilometer gibt oder nicht.
Dies kann er in *O(P(n))* entscheiden, wobei *P(n)* eine polynomielle Funktion der Anzahl der Orte `n` ist.

Jetzt wollen wir daraus die kürzeste Tour finden (`Min(b)`), für das Rudolph sagt, dass es eine Tour gibt.
Da Rudolph in polynomieller Zeit entscheiden kann, ob es eine Tour gibt oder nicht, können wir die binäre Suche verwenden, um die kürzeste Tour zu finden.
Die binäre Suche hat eine Laufzeit von *O(log B * P(n))*, wobei `B` die maximale mögliche Tourlänge ist.

Letzendlich wollen wir noch die konkrete Tour finden.
Nutzen wir dafür folgenden ALgorithmus:
Nehme an, dass die Distanzen zwischen den Orten ganzzahlig sind.
Überprüfe solange ob eine Kante zur Route gehört, bis wir einen Zyklus haben.
Herausfinden, ob eine Kante zur Route gehört, können wir, indem wir einer Kante einen Wert `+1` zuweisen.
Wenn die Kante zur Route gehört, dann wird Rudolph sagen, dass es nur eine Tour der Länge `Min(b)+1` gibt.
Wenn die Kante nicht zur Route gehört, dann wird Rudolph sagen, dass es eine Tour der Länge `Min(b)` gibt.
Dadurch können wir die kürzeste Tour finden.
Ein vollständiger Graph hat `n*(n-1)/2` Kanten, wobei `n` die Anzahl der Orte ist.

Die Gesamtlaufzeit der Bestimmung der konkreten Tour beträgt daher *O((log(B) \* P(n)) + (n\*(n-1)/2 \* P(n))) = O(n^2\*P(n))*.

Wir haben also einen Algorithmus, der anhand der Entscheidungsvariante des TSP in polynomieller Zeit die kürzeste Tour finden kann.
Daher ist TSP in P, wenn die Entscheidungsvariante des TSP in P ist.