#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
from fractions import Fraction
from sympy import *
from sympy.utilities.iterables import default_sort_key


# In[ ]:


def input_d():
  f=input('Введите функцию для решения задачи максимизации в аналитическом виде. Пример: x1+x2. Ввод:')
  g_count=input('Введите количество ограничений типа равенств. Пример: 2. Ввод:')
  g1=[]
  for i in range(int(g_count)):
    s='Введите ограничение f(x,y,z,...)=0. Пример: x1+x2-4. Ввод '+str(i+1)+':'
    g_=input(s)
    g1.append(g_)
  g_count=input('Введите количество ограничений типа неравенств. Пример: 2. Ввод:')
  g2=[]
  for i in range(int(g_count)):
    s='Введите ограничение f(x,y,z,...)<=0. Пример: x1+x2-4. Ввод '+str(i+1)+':'
    g_=input(s)
    g2.append(g_)
  return [f,g1,g2]


# In[ ]:


def processing(s):
  f=parsing.sympy_parser.parse_expr(s[0])
  list_f=list(f.free_symbols)
  for g in s[1]:
    g=parsing.sympy_parser.parse_expr(g)
    list_f=list(set(list_f+list(g.free_symbols)))
  for g in s[2]:
    g=parsing.sympy_parser.parse_expr(g)
    list_f=list(set(list_f+list(g.free_symbols)))
  c=[]
  list_f.sort(key = default_sort_key)
  for x in list_f:
    c.append(float(f.diff(x)))
  c=c+[0]*len(s[2])
  A=[]
  b=[]
  for g in s[1]:
    g=parsing.sympy_parser.parse_expr(g)
    g_list=[]
    for x in list_f:
      g_list.append(float(g.diff(x)))
    const = float(g.func(*[term for term in g.args if not term.free_symbols]))
    g_list=g_list+[0]*len(s[2])
    A.append(g_list)
    b.append(-const)
  k=len(list_f)
  for g in s[2]:
    g=parsing.sympy_parser.parse_expr(g)
    g_list=[]
    for x in list_f:
      g_list.append(float(g.diff(x)))
    const = float(g.func(*[term for term in g.args if not term.free_symbols]))
    g_list=g_list+[0]*len(s[2])
    g_list[k]=1
    k+=1
    A.append(g_list)
    b.append(-const)
  return c,A,b,f,list_f


# In[ ]:


def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]


# In[ ]:


def can_be_improved(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])


# In[ ]:



def get_pivot_position(tableau):
    z = tableau[-1]
    column = next(i for i, x in enumerate(z[:-1]) if x > 0)
    
    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)
        
    if (all([r == math.inf for r in restrictions])):
        print("Решение отсутствует.")
        return 0

    row = restrictions.index(min(restrictions))
    return row, column


# In[ ]:


def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]
    
    i, j = pivot_position
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value
    
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier
   
    return new_tableau


# In[ ]:


def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

def get_solution(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)
        
    return solutions


# In[ ]:


def simplex(c, A, b):
    tableau = to_tableau(c, A, b)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    return get_solution(tableau), tableau


# In[ ]:


def gomori(c, A, b, f, list_f):
  solution, tableau = simplex(c, A, b)
  while len([c for c in solution if c%1==0]) != len(solution):
    remain=[]
    for sol in solution:
      remain.append(Fraction(sol).limit_denominator(1000)%1)
    rem_max=remain.index(max(remain))
    k=0
    for tab in tableau[:-1]:
      if solution[rem_max] in tab:
        tab_max=k
        break
      k+=1
    t_list=[]
    for t in tableau[tab_max][:-1]:
      if t<0:
        t_list.append(-(t-(int(t)-1)))
      if t>0:
        t_list.append(-(t-int(t)))
      if t==0:
        t_list.append(0)
    t_list.append(tableau[tab_max][-1]-int(tableau[tab_max][-1]))
    for i in range(len(tableau)):
      tableau[i]=np.array(list(tableau[i][:-1])+[0]+[tableau[i][-1]])
    a=tableau[-1]
    tableau[-1]=(np.array(list(t_list[:-1])+[1]+[-t_list[-1]]))
    tableau.append(a)
    tableau[-1][-1]=-tableau[-1][-1]
    teta=[]
    for i in range(len(tableau[-1])):
      if tableau[-2][i]==0:
        teta.append(10*10)
      else:
        teta.append(tableau[-1][i]/tableau[-2][i])
    teta_min=teta.index(min(teta[:-2]))
    tableau = pivot_step(tableau, (len(tableau)-2,teta_min))
    solution = get_solution(tableau)
    for i in range(len(solution)):
      solution[i]=round(solution[i],5)
  for i in range(len(solution)):
      solution[i]=round(solution[i])
  d_solve=dict(zip(list_f,solution[:len(list_f)]))
  d_solve_new=dict()
  for a in d_solve:
    if a in f.free_symbols:
      d_solve_new[a]=d_solve[a]
  print('Решением методом Гомори',d_solve_new,'max f =',f.subs(d_solve_new))


# In[ ]:


def all_f_gomori():
  s=input_d()
  d=processing(s)
  gomori(*d)


# In[ ]:


all_f_gomori()


# In[ ]:


def branches_and_bound(c, A, b, last_sol):
  solution, tableau = simplex(c, A, b)
  sol=solution
  if solution==last_sol:
    return 'Ветка не подходит'
  else:
    for i in range(sum(np.array(c)!=0)):
      if solution[i]%1!=0:
          A_new1=A.copy()
          b_new1=b.copy()
          A_new2=A.copy()
          b_new2=b.copy()
          s1=[0]*len(solution)
          s2=[0]*len(solution)
          sol1=solution.copy()
          sol2=solution.copy()
          sol1[i]=math.trunc(sol1[i])
          sol2[i]=-(math.trunc(sol2[i])+1)
          s1[i]=1
          s2[i]=-1
          A_new1.append(s1)
          b_new1.append(sol1[i])
          A_new2.append(s2)
          b_new2.append(sol2[i])
          sol = [branches_and_bound(c, A_new1, b_new1, solution),branches_and_bound(c, A_new2, b_new2, solution)]
    return sol


# In[ ]:


def unwrap_list(mylist, result):
   if any(isinstance(i, list) for i in mylist):
      for value in mylist:
         unwrap_list(value, result)
   else:
      result.append(mylist)


# In[ ]:


def find_good(c,A,b,f,list_f,last_sol=[0]):
  solution = branches_and_bound(c,A,b,last_sol)
  result = []
  unwrap_list(solution, result)
  d=[]
  for sol in result:
    if sol!='Ветка не подходит':
      d.append(np.sum(np.array(c)*np.array(sol)))
  d_max=np.argmax(d)
  result=result[d_max]
  d_solve=dict(zip(list_f,result[:len(list_f)]))
  d_solve_new=dict()
  for a in d_solve:
    if a in f.free_symbols:
      d_solve_new[a]=d_solve[a]
  print('Решением методом ветвей и границ',d_solve_new,'max f =',f.subs(d_solve_new))


# In[ ]:


def all_f_branches_and_bound():
  s=input_d()
  d=processing(s)
  find_good(*d)


# In[ ]:


all_f_branches_and_bound()


# In[ ]:




