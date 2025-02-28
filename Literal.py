class Literal:
    """
    A literal is a variable or its negation.
    :param name: Name of the variable
    :param negated: Boolean indicating if the literal is negated
    :return: Literal, which is a variable or its negation
    """
    def __init__(self, name: str, negated: bool = False):
        self.name = name
        self.negated = negated

    def __neg__(self):
        """Returns the negation of the literal."""
        return Literal(self.name, not self.negated)

    def __eq__(self, other):
        """Checks if two literals are equal."""
        return isinstance(other, Literal) and self.name == other.name and self.negated == other.negated

    def __hash__(self):
        """Returns the hash of the literal for set operations."""
        return hash((self.name, self.negated))

    def __repr__(self):
        """Returns the string representation of the literal."""
        return f"~{self.name}" if self.negated else self.name

    def evaluate(self, assignment: dict) -> bool:
        """Evaluates the literal given a variable assignment."""
        value = assignment.get(self.name, False)
        return not value if self.negated else value
    
class Clause:
    """
    A clause is a disjunction of literals.
    :param literals: Set of literals
    :return: Clause, which is a disjunction of literals
    """
    def __init__(self, literals: set):
        self.literals = literals

    def __repr__(self):
        """Returns the string representation of the clause."""
        return f"{' | '.join([str(literal) for literal in self.literals])}"

    def __eq__(self, other):
        """Checks if two clauses are equal."""
        return isinstance(other, Clause) and self.literals == other.literals

    def __hash__(self):
        """Returns the hash of the clause for set operations."""
        return hash(frozenset(self.literals))

    def evaluate(self, assignment: dict) -> bool:
        """Evaluates the clause given a variable assignment."""
        return any([literal.evaluate(assignment) for literal in self.literals])

class CNF:
    """
    Conjunctive Normal Form (CNF) is a conjunction of disjunctions of literals.
    :param clauses: Set of clauses, where each clause is a disjunction of literals
    :return: CNF formula, which is a conjunction of clauses
    """
    
    def __init__(self, clauses: set):
        self.clauses = clauses

    def __repr__(self):
        """Returns the string representation of the CNF."""
        return f"{' & '.join([str(clause) for clause in self.clauses])}"

    def __eq__(self, other):
        """Checks if two CNFs are equal."""
        return isinstance(other, CNF) and self.clauses == other.clauses

    def __hash__(self):
        """Returns the hash of the CNF for set operations."""
        return hash(frozenset(self.clauses))

    def evaluate(self, assignment: dict) -> bool:
        """Evaluates the CNF given a variable assignment."""
        return all([clause.evaluate(assignment) for clause in self.clauses])
    
# # from Literal import Literal

# Erstellen von Literalen
# a = Literal("a")
# not_a = -a
# b = Literal("b")

# # Erstellen einer Variablenzuweisung
# assignment = {"a": True, "b": False}

# # Bewertung der Literale
# print(a.evaluate(assignment))  # True
# print(not_a.evaluate(assignment))  # False
# print(b.evaluate(assignment))  # False
# print((-b).evaluate(assignment))  # True

# abc = Clause({a, b, -a})
# print(abc.evaluate(assignment))  # True

# cnf = CNF({abc, Clause({a, -b}), Clause({-a, b})})
# print(cnf.evaluate(assignment))  # True

# # print names of literals, clauses and cnf
# print(a)
# print(abc)
# print(cnf)

