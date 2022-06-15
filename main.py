import numpy as np
from sympy import *
from scipy.optimize import linprog
from copy import deepcopy

def branche_boundaries(F: str, constraint: list):
    '''
    inputs:
        F: function as a string (example: '2*x1 + 6*x2')
        constraint: restrictions in the form of a list of strings (example: ['x1 - x2 <= 4', '-3*x1 <= 17'])
    outputs:
        tuple: (point, result of function);
        example: (array([1., 5.]), 4.00000000000000)
    '''
    
    exp = sympify(F)
    
    count_constraint = int(input('Введите количество ограничений: '))
    
    symbs = sorted(list(exp.free_symbols), key=lambda x: str(x))
    
    c = [float(exp.coeff(symb)) for symb in symbs]
    
    dic_constr = {i: [] for i in range(1, count_constraint+1)}
    for i in range(len(constraint)):
        ineq = sympify(constraint[i])
        ex = ineq.args[0]
        dic_constr[i+1] = [float(ex.coeff(symb)) for symb in symbs]

    A = [v for v in dic_constr.values()]
    
    for i in range(len(symbs)):
        lst = [0.0]*len(symbs)
        lst[i] = -1.0
        A.append(lst)
    
    b = []
    for ineq in constraint:
        ine = sympify(ineq)
        b.append(ine.args[1])
    
    b += [0]*len(symbs)
    
    res = linprog(c, A_ub=A, b_ub=b, method='simplex')['x']
    
    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res
    
    try:
        res_zlp1 = ZLP1(F, constraint, res, symbs, A, b, c)
    except:
        res_zlp1 = None
        
    try:
        res_zlp2 = ZLP2(F, constraint, res, symbs, A, b, c)
    except:
        res_zlp2 = None
    
    if res_zlp1 is not None and res_zlp2 is not None:
        if exp.subs(list(zip(symbs, res_zlp1))) <= exp.subs(list(zip(symbs, res_zlp2))):
            return res_zlp1, exp.subs(list(zip(symbs, res_zlp1)))
        else:
            return res_zlp2, exp.subs(list(zip(symbs, res_zlp2)))
    elif res_zlp1 is not None and res_zlp2 is None:
        return res_zlp1, exp.subs(list(zip(symbs, res_zlp1)))
    elif res_zlp1 is None and res_zlp2 is not None:
        return res_zlp2, exp.subs(list(zip(symbs, res_zlp2)))
    else:
        return None
    
def ZLP1(func, constraint, res_last, symbs, A, b, c):
    i = np.argmax(res_last%1)
    whole = int(res_last[i])
    lst = [0]*len(symbs)
    lst[i] = 1
    
    
    A_new = deepcopy(A)
    b_new = deepcopy(b)
    A_new.append(lst)
    b_new.append(whole)

    res = linprog(c, A_ub=A_new, b_ub=b_new, method='simplex')['x']
    
    
    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res
    
    res_zlp3 = ZLP1(func, constraint, res, symbs, A_new, b_new, c)
    res_zlp4 = ZLP2(func, constraint, res, symbs, A_new, b_new, c)
    
    
    to_return = []
    if all(str(i).split('.')[-1] == '0' for i in res_zlp3) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp3))) for const in constraint]):
        to_return.append(res_zlp3)
    if all(str(i).split('.')[-1] == '0' for i in res_zlp4) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp4))) for const in constraint]):
        to_return.append(res_zlp4)
        
    if to_return:
        return to_return
    else:
        return None

def ZLP2(func, constraint, res_last, symbs, A, b, c):
    i = np.argmax(res_last%1)
    whole = -int(res_last[i])-1 
    lst = [0]*len(symbs)
    lst[i] = 1
    
    A_new = deepcopy(A)
    b_new = deepcopy(b)
    A_new.append(lst)
    b_new.append(whole)

    res = linprog(c, A_ub=A_new, b_ub=b_new, method='simplex')['x']
    
    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res
    
    res_zlp5 = ZLP1(func, constraint, res, symbs, A_new, b_new, c)
    res_zlp6 = ZLP2(func, constraint, res, symbs, A_new, b_new, c)
    
    to_return = []
    if all(str(i).split('.')[-1] == '0' for i in res_zlp5) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp5))) for const in constraint]):
        to_return.append(res_zlp5)
    if all(str(i).split('.')[-1] == '0' for i in res_zlp5) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp5))) for const in constraint]):
        to_return.append(res_zlp6)
        
    if to_return:
        return to_return
    else:
        return None

 
