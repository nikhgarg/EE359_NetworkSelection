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
    
test_calc_correlated()