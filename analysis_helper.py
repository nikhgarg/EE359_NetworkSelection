# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:49:59 2015

@author: nikhil
"""
import environment
import plotly.plotly as py
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import prettyplotlib as ppl


#def case0values():
#    calculatecapacity(UE, BS, N_connected, variables)
#    B = variables['B']
#    No = variables['No']
#    Pt = variables['P']
#    K = variables['K_PL']
#    alpha = variables['alpha']

#	\item Find any dominant strategies, and corresponding best-response strategies. 
#	\item If no dominant strategies exist, find the mixed strategy in which each UE connects to each BS with nonzero probability and calculate the corresponding log sum rate.
#	\item For each of the 2 pure strategies, calculate the log sum rate.
#	\item Choose the strategy that maximizes the log sum rate.

def findDominantStrategies(C, Kcoex):
    UE0dominant = -1
    UE1dominant = -1
    #if 0 has dominant strategy, find 1's best response to that strategy
    #if 0 does not have dominant, check if 1 does
        #if 1 does, find 0's best response
        #if 1 doesn't indicate that there are no dominant strategies.
            #this case should correspond to none found earlier...
        
    if 1/2*Kcoex*C[0,0] > (1-Kcoex)*C[0,1]: # BS0 dominant when 1/2*Kcoex*C00 > (1-K)C01
        UE0dominant = 0
        if 1/2*Kcoex*C[1,0] > (1-Kcoex)*C[1,1]: #best response to BS0
            UE1dominant = 0
        else:
            UE1dominant = 1
        return [UE0dominant, UE1dominant]
    elif (1-Kcoex)/2*C[0,1] > Kcoex*C[0,0]:         # BS1 dominant when (1-K)/2*C01 > KC00
        UE0dominant = 1
        if Kcoex*C[1,0] > (1-Kcoex)/2*C[1,1]:
            UE1dominant = 0
        else:
            UE1dominant = 1
        return [UE0dominant, UE1dominant]
    elif 1/2*Kcoex*C[1,0] > (1-Kcoex)*C[1,1]: #UE1 has BS0 dominant
        UE1dominant = 0
        if 1/2*Kcoex*C[0,0] > (1-Kcoex)*C[0,1]: #UE0's best response to BS0
            UE0dominant = 0
        else:
            UE0dominant = 1
        return [UE0dominant, UE1dominant]
    elif (1-Kcoex)/2*C[1,1] > Kcoex*C[1,0]:         # 
        UE1dominant = 1
        if Kcoex*C[0,0] > (1-Kcoex)/2*C[0,1]:
            UE0dominant = 0
        else:
            UE0dominant = 1
        return [UE0dominant, UE1dominant]
    return None
    
def findTrueMixedStrategy(C, Kcoex):
    a = 1/2*Kcoex*C[1,0] - Kcoex*C[1,0]    
    c = Kcoex*C[1,0]
    b = (1-Kcoex)*C[1,1] - (1-Kcoex)/2*C[1,1]
    d = (1-Kcoex)/2*C[1,1]
    mixed0 = [(d-c)/(a-b), 1-(d-c)/(a-b)]
    
    a = 1/2*Kcoex*C[0,0] - Kcoex*C[0,0]    
    c = Kcoex*C[0,0]
    b = (1-Kcoex)*C[0,1] - (1-Kcoex)/2*C[0,1]
    d = (1-Kcoex)/2*C[0,1]
    mixed1 = [(d-c)/(a-b), 1-(d-c)/(a-b)]
    
    return [mixed0, mixed1];

def findPossibleTrueStrategies(C, Kcoex):
    pure1 = [[1, 0], [1, 0]]
    pure2 = [[0, 1], [1, 0]]
    pure3 = [[1, 0], [0, 1]]
    pure4 = [[0, 1], [0, 1]]
    return [pure1, pure2, pure3, pure4]
    
def findBestMixedStrategy(C, Kcoex):
    strategies = findPossibleTrueStrategies(C, Kcoex)
    strategies.append( findTrueMixedStrategy(C, Kcoex))
    utils = np.zeros(5)
    for i in range(0, 5):
        utils[i] = calculateLogSumUtilitiesFromMixedStrategies(strategies[i], Kcoex, C)
    return strategies[np.argmax(utils)]        
    
def calcUtilityFromStrategy(userind, strategy, Kcoex, A):
    u = A[userind,0]*Kcoex*strategy[userind][0]*(strategy[1-userind][0]/2 + strategy[1-userind][1]) + A[userind,1]*(1-Kcoex)*strategy[userind][1]*(strategy[1-userind][0] + strategy[1-userind][1]/2)
    return u

def calculateLogSumUtilitiesFromMixedStrategies(strategy, Kcoex, A):
    u0 = calcUtilityFromStrategy(0, strategy, Kcoex, A)
    u1 = calcUtilityFromStrategy(1, strategy, Kcoex, A)
    return np.log2(u0) + np.log2(u1)
    
def testfindMixedStrategies():
    A = np.matrix([[np.random.rand(), np.random.rand()], [np.random.rand(), np.random.rand()]])
 #   A = np.matrix([[3,1], [4,2]])
    K = .5
    print(findBestMixedStrategy(A, K))
    #A = np.matrix([[.1,0], [.1,.1]])
################################################
def calc_correlated_equil(C):
    minp = max(1/2*C[0,1]/(C[0,0] + 1/2*C[0,1]), 1/2*C[1,1]/(C[1,0] + 1/2*C[1,1]))
    maxp = min(C[0,1]/(1/2*C[0,0] + C[0,1]), C[1,1]/(1/2*C[1,0] + C[1,1]))
    if maxp < minp: #correlated equilibrium not working. Defaulting to their dominant strategies
        return -1
    c1 = C[0,0]
    c2 = C[0,1]
    c3 = C[1,0]
    c4 = C[1,1]
    a = c2*c3 + c1*c4 - 2*c2*c4
    b = c1*c3 - c2*c3 - c1*c4 + c2*c4
    maximing = -a/(2.0*b)
    if maximing < minp or maximing > maxp: #maximizing is out of range, boundry value is appropriate
        minv = np.matrix([[minp], [1-minp]])
        Umin = C*minv
        minUprod = Umin[0]*Umin[1]
        
        maxv = np.matrix([[maxp], [1-maxp]])
        Umax = C*maxv
        maxUprod = Umax[0]*Umax[1]

        if minUprod > maxUprod:
            return minp
        else:
            return maxp
    else:
        return maximing
        
    
def test_calc_correlated():
    A = np.matrix([[np.random.rand(), np.random.rand()], [np.random.rand(), np.random.rand()]])
    #A = np.matrix([[.1,0], [.1,.1]])

    pstar = -1
    maxval = -1
    minp = max(1/2*A[0,1]/(A[0,0] + 1/2*A[0,1]), 1/2*A[1,1]/(A[1,0] + 1/2*A[1,1]))
    maxp = min(A[0,1]/(1/2*A[0,0] + A[0,1]), A[1,1]/(1/2*A[1,0] + A[1,1]))
    for p in np.arange(minp,maxp, .0001):
        pm = np.matrix([[p], [1-p]])
        U = A*pm
        val = U[0]*U[1]
        if val > maxval:
            maxval= val
            pstar = p
            
    print(minp, maxp, pstar)
    print(calc_correlated_equil(A, minp, maxp));

def cap_simp(d, alpha = 3):
    return np.log(1 + d^(-alpha), 2)

def Kheatmap():
    xloc = np.arange(-1.5,1.5,.005)
    yloc = np.arange(-1.5,1.5,.005)
    K = np.zeros((len(xloc), len(yloc))    )
    BS1 = [ 0, -1]
    BS2 = [0, 1]
    
    
    for y in range(0, len(yloc)):
        for x in range(0, len(xloc)):
            c1 = np.log2(1 + np.power(environment.distanceloc(BS1, [xloc[x], yloc[y]]), -3))
            c2 = np.log2(1 + np.power(environment.distanceloc(BS2, [xloc[x], yloc[y]]), -3))
            K[x,y] = c2/(c1 + c2)
      
   # plt.clf()
   # plt.imshow(K, extent=extent)
   # plt.show()
    plt.pcolor(xloc, yloc, K)
    plt.colorbar()
    plt.title("K value at transition")
#Kheatmap()  
#test_calc_correlated()

testfindMixedStrategies()