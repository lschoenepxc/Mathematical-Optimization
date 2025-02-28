# https://github.com/asgordon/DPLL/blob/master/README.md
# Andrew Gordon
# Feb 19, 2015
# Revised June 19, 2015 for better input/output

def standardize(cnf):
    if type(cnf) is str: # must be a single positive literal
        return ["and", ["or", cnf]]
    elif cnf[0] == "not": # must be a single negative literal
        return ["and", ["or", cnf]]
    elif cnf[0] == "or": # a single clause
        return ["and", cnf]
    else:
        result = ["and"]
        for c in cnf[1:]:
            if type(c) == str:
                result.append(["or", c])
            elif c[0] == "not":
                result.append(["or", c])
            else:
                result.append(c)
        return result

def allTrue(cnf, model): # at least one member of model in each clause
    for clause in cnf[1:]: # skip the "and"
        if len([var for var in clause[1:] if var in model]) == 0:
            return False
    return True

def compliments(model): # returns the compliment of each model literal
    result = []
    for literal in model:
        if type(literal) is str:
            result.append(["not", literal])
        else:
            result.append(literal[1])
    return result

def someFalse(cnf, model): # some clause cannot be satisfied
    modelCompliments = compliments(model)
    for clause in cnf[1:]:
        if len([var for var in clause[1:] if var not in modelCompliments]) == 0:
            return True
    return False

def pureLiteral(cnf, model): # finds 1 pure literal not already in model
    modelCompliments = compliments(model)
    candidates = []
    for clause in cnf[1:]:
        if len([var for var in clause[1:] if var in model]) == 0:
            # clause not yet satisfied by model
            candidates = candidates + [var for var in clause[1:]]
    candidateCompliments = compliments(candidates)
    pure = [var for var in candidates if var not in candidateCompliments]
    for var in pure:
        if var not in model and var not in modelCompliments:
            return var
    return False

def unitClause(cnf, model): # finds 1 literal not in model appearing by itself in a clause
    modelCompliments = compliments(model)
    for clause in cnf[1:]:
        remaining = [var for var in clause[1:] if var not in modelCompliments]
        if len(remaining) == 1:
            if remaining[0] not in model:
                return remaining[0]
    return False

def pickSymbol(cnf, model): # finds a positive literal not in model or model compliments
    combined = model + compliments(model)
    for clause in cnf[1:]:
        for literal in clause[1:]:
            if type(literal) is str and literal not in combined:
                return literal
    return False

def dpll(cnf):
    #print "starting", standardize(cnf)
    return dpll1(standardize(cnf), [])

def dpll1(cnf, model):
    if allTrue(cnf, model):
        return model
    if someFalse(cnf, model):
        return False
    pure = pureLiteral(cnf, model)
    if pure:
        return dpll1(cnf, model + [pure])
    unit = unitClause(cnf, model)
    if unit:
        return dpll1(cnf, model + [unit])
    pick = pickSymbol(cnf, model)
    if pick:
        # try positive
        result = dpll1(cnf, model + [pick])
        if result:
            return result
        else:
            # try negative
            result = dpll1(cnf, model + [['not', pick]])
            if result:
                return result
            else:
                return False

def formatOutput(result):
    if result == False:
        return ["false"]
    else:
        mod = ["true"]
        for v in result:
            if type(v) is str:
                mod.append(v + "=true")
            else:
                mod.append(v[1] + "=false")
        return mod
            
# test examples from lecture

# ex1 = ['and',
#        ['or', 'P', 'Q'],
#        ['or', ['not', 'P'], 'R'],
#        ['or', ['not', 'R'], ['not', 'P']],
#        ['or', ['not', 'Q'], 'S', ['not', 'T']],
#        ['or', 'T']]

# ex2 = ['and',
#        ['or', ['not', 'P'], 'Q'],
#        ['or', ['not', 'Q'], ['not', 'P']],
#        ['or', 'P', ['not', 'Q']]]
        
# print(dpll(ex1))
# print(dpll(ex2))