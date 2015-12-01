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
import math

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

def findPossibleTrueStrategies():
    pure1 = [[1, 0], [1, 0]]
    pure2 = [[0, 1], [1, 0]]
    pure3 = [[1, 0], [0, 1]]
    pure4 = [[0, 1], [0, 1]]
    return [pure1, pure2, pure3, pure4]
    
def findBestMixedStrategy(C, Kcoex):
    strategies = findPossibleTrueStrategies()
    strategies.append( findTrueMixedStrategy(C, Kcoex))
    utils = np.zeros(5)
    for i in range(0, 5):
        utils[i] = calculateLogSumUtilitiesFromMixedStrategies(strategies[i], Kcoex, C)
#    print(strategies, utils)
    return strategies[np.argmax(utils)]        
    
def calcUtilityFromStrategy(userind, strategy, Kcoex, A):
    u = A[userind,0]*Kcoex*strategy[userind][0]*(strategy[1-userind][0]/2 + strategy[1-userind][1]) + A[userind,1]*(1-Kcoex)*strategy[userind][1]*(strategy[1-userind][0] + strategy[1-userind][1]/2)
    return u

def calculateLogSumUtilitiesFromMixedStrategies(strategy, Kcoex, A):
    u0 = calcUtilityFromStrategy(0, strategy, Kcoex, A)
    u1 = calcUtilityFromStrategy(1, strategy, Kcoex, A)
    return np.log2(u0) + np.log2(u1)
 #   return u0 + u1

def testfindMixedStrategies():
    A = np.matrix([[np.random.rand(), np.random.rand()], [np.random.rand(), np.random.rand()]])
 #   A = np.matrix([[3,1], [4,2]])
    K = .5
#    print(findBestMixedStrategy(A, K))
    #A = np.matrix([[.1,0], [.1,.1]])
################################################
def calc_correlated_equil(A, K):
    C = np.zeros((2, 2))
    C[0,0] = K*A[0,0]
    C[0,1] = (1-K)*A[0,1]
    C[1,0] = (1-K)*A[1,1]
    C[1,1] = K*A[1,0]
    epsilon = .01 #fix rounding errors...
    minp = max(1/2*C[0,1]/(C[0,0] + 1/2*C[0,1]), 1/2*C[1,1]/(C[1,0] + 1/2*C[1,1]))
    maxp = min(C[0,1]/(1/2*C[0,0] + C[0,1]), C[1,1]/(1/2*C[1,0] + C[1,1]))
    if maxp + epsilon < minp - epsilon: #correlated equilibrium not working. Defaulting to their dominant strategies
        strats = findPossibleTrueStrategies()
        utils = np.zeros(4)
        for i in range(0, 4):
            utils[i] = calculateLogSumUtilitiesFromMixedStrategies(strats[i], K, A)
        strat = strats[np.argmax(utils)]
        return False, strat, [calcUtilityFromStrategy(0, strat, K, A), calcUtilityFromStrategy(1, strat, K, A)]
    c1 = C[0,0]
    c2 = C[0,1]
    c3 = C[1,0]
    c4 = C[1,1]
#    print(c1, c2, c3, c4)
    a = c2*c3 + c1*c4 - 2*c2*c4
    b = c1*c3 - c2*c3 - c1*c4 + c2*c4
    maximing = -a/(2.0*b)
    if math.isnan(maximing):
        maximing = .5
    Umaximing = C*np.matrix([[maximing], [1-maximing]])
#    print([Umaximing[0,0], Umaximing[1,0]])
    if maximing < minp - epsilon or maximing > maxp + epsilon: #maximizing is out of range, boundry value is appropriate
        minv = np.matrix([[minp], [1-minp]])
        Umin = C*minv
        minUprod = Umin[0]*Umin[1]
        
        maxv = np.matrix([[maxp], [1-maxp]])
        Umax = C*maxv
        maxUprod = Umax[0]*Umax[1]

        if minUprod > maxUprod:
            return True, minp, [Umin[0,0], Umin[1,0]]
        else:
            return True, maxp, [Umax[0,0], Umax[1,0]]
    else:
        return True, maximing, [Umaximing[0,0], Umaximing[1,0]] #awkward thing to flatten matrix
        
    
def test_calc_correlated():
    #A = np.matrix([[np.random.rand(), np.random.rand()], [np.random.rand(), np.random.rand()]])
    A = np.matrix([[1,1], [1, 1]])
    Kcoex = .666666666667
    K = Kcoex
    C = np.zeros((2, 2))
    C[0,0] = K*A[0,0]
    C[0,1] = (1-K)*A[0,1]
    C[1,0] = (1-K)*A[1,1]
    C[1,1] = K*A[1,0]
    pstar = -1
    maxval = np.matrix(1)
    minp = max(1/2*C[0,1]/(C[0,0] + 1/2*C[0,1]), 1/2*C[1,1]/(C[1,0] + 1/2*C[1,1]))
    maxp = min(C[0,1]/(1/2*C[0,0] + C[0,1]), C[1,1]/(1/2*C[1,0] + C[1,1]))
    for p in np.arange(minp,maxp, .001):
        pm = np.matrix([[p], [1-p]])
        U = C*pm
        val = U[0]*U[1]
 #       print (p, val)
        if val > maxval:
            maxval= val
            pstar = p
            
    print(maxval[0,0], pstar, minp, maxp)
    found, maxp, u = calc_correlated_equil(A, Kcoex)
    print(u[0]*u[1], maxp, found);

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

#testfindMixedStrategies()