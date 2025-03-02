from Literal import Literal, Clause, CNF
from typing import List, Dict, Tuple, Any

class DPLL(object):
    
    def __init__(self):
        super().__init__()
        
    def unit_propagate(self, cnf: CNF, assignment: dict) -> CNF:
        """
        Unit propagation simplifies the CNF by assigning a value to a unit clause. (unit clause = clause with only one literal)
        :param cnf: CNF formula
        :param assignment: Dictionary of variable assignments
        :return: Simplified CNF formula
        """
        new_clauses = set(cnf.clauses)
        unit_clauses = [clause for clause in cnf.clauses if len(clause.literals) == 1]
        
        while unit_clauses:
            unit_clause = unit_clauses.pop()
            unit_literal = next(iter(unit_clause.literals))
            assignment[unit_literal.name] = not unit_literal.negated
            
            new_clauses = {clause for clause in new_clauses if unit_literal not in clause.literals}
            for clause in new_clauses:
                if -unit_literal in clause.literals:
                    new_literals = clause.literals - {-unit_literal}
                    new_clause = Clause(new_literals)
                    new_clauses.remove(clause)
                    new_clauses.add(new_clause)
                    if len(new_clause.literals) == 1:
                        unit_clauses.append(new_clause)
        
        return CNF(new_clauses)
    
    def pure_literal_assign(self, cnf: CNF, assignment: dict) -> CNF:
        """
        Pure literal elimination assigns a value to a pure literal. (pure literal = literal that only occurs with one polarity)
        :param cnf: CNF formula
        :param assignment: Dictionary of variable assignments
        :return: Simplified CNF formula
        """
        literals = {literal for clause in cnf.clauses for literal in clause.literals}
        pure_literals = {literal for literal in literals if -literal not in literals}
        
        new_clauses = set(cnf.clauses)
        for pure_literal in pure_literals:
            assignment[pure_literal.name] = not pure_literal.negated
            new_clauses = {clause for clause in new_clauses if pure_literal not in clause.literals}
        return CNF(new_clauses)
        
    
    def dpll(self, cnf: CNF, assignment: dict) -> Tuple[bool, dict]:
        """
        DPLL algorithm to determine if a CNF formula is satisfiable.
        :param cnf: CNF formula
        :param assignment: Dictionary of variable assignments
        :return: Boolean indicating if the formula is satisfiable, and the assignment
        Complexity: O(2^n), where n is the number of variables, but better with unit propagation and pure literal elimination (implemented)
        """
        # Unit propagation
        cnf = self.unit_propagate(cnf, assignment)
        
        # Pure literal elimination
        cnf = self.pure_literal_assign(cnf, assignment)
        
        # Stopping conditions
        # if cnf empty
        if not cnf.clauses:
            return True, assignment
        # if cnf has empty clause
        if any(not clause.literals for clause in cnf.clauses):
            return False, {}
        
        # Choose a literal
        unassigned_literals = {literal for clause in cnf.clauses for literal in clause.literals if literal.name not in assignment}
        if not unassigned_literals:
            # Check if the current assignment satisfies the CNF
            if cnf.evaluate(assignment):
                return True, assignment
            return False, {}
        
        chosen_literal = next(iter(unassigned_literals))
        
        # Try assigning the literal to True
        assignment[chosen_literal.name] = not chosen_literal.negated
        satisfiable, result = self.dpll(cnf, assignment)
        if satisfiable:
            # Check if the current assignment satisfies the CNF
            if cnf.evaluate(result):
                return True, result

        # Backtrack and try assigning the literal to False
        assignment[chosen_literal.name] = chosen_literal.negated
        satisfiable, result = self.dpll(cnf, assignment)
        if satisfiable:
            # Check if the current assignment satisfies the CNF
            if cnf.evaluate(result):
                return True, result

        # If neither assignment works, backtrack
        del assignment[chosen_literal.name]
        return False, {}