# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:51:18 2018

@author: olmer.garciab
"""

# implementing class example

P=[[0,0.5,0,0,0.,0.5,0],
   [0,0,0.8,0,0.0,0,0.2],
   [0,0.,0,0.6,0.4,0,0],
   [0,0.,0,0,0.,0,1],
   [0.2,0.4,0.4,0,0,0,0],
   [0.1,0.,0,0,0.,0.9,0],
   [0,0.,0,0,0.,0,1]]
R=[-2,-2,-2,10,1,-1,0]


# total dicount reward
def G_t(S,R,gamma):
    g=0
    for s,i in zip(S,range(len(S))): 
        g=g+R[s]*gamma**i  
    return g

#g = lambda y: sum(  f(y) for f in (lambda x: x**i for i in range(n))  )
# for example for the chain of state S_1
gamma=0.5
S_1=[0,1,2,3,6]
print(G_t(S_1,R,gamma))




#dynamic programming
#based in #https://harderchoices.com/2018/02/26/dynamic-programming-in-python-reinforcement-learning/
def iterative_value_function(N, theta=0.0001, gamma=0.9):
    V_s =R.copy() # 1.
    probablitiy_map = P # 2.
    delta = 100 # 3.
    while not delta < theta: # 4.
        delta = 0 # 5.
        for state in range(0,N): # 6.
            v = V_s[state] # 7.
            
            total =R[state] # 8.
            for state_prime in range(0,N):
                total += probablitiy_map[state][state_prime] * (gamma * V_s[state_prime])
                #print(total)
                
            V_s[state] =total # 9.
            delta = max(delta, abs(v - V_s[state])) # 10.
            #print(delta)
    V_s=[round(v,2) for v in V_s]
    return V_s # 11.


N=len(R)




print('gamma',0.9,iterative_value_function(N,gamma=0.9))
print('gamma',1,iterative_value_function(N,gamma=1))
print('gamma',0,iterative_value_function(N,gamma=0))


#vectorial way
import numpy as np
from numpy.linalg import inv
gamma=0.9
P=np.array(P)
R=np.array(R).reshape((-1,1))
v=np.matmul(inv(np.eye(N)-gamma*P),R)
print(v)
