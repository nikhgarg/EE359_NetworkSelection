# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:20:10 2015

@author: Nikhil
"""
import math
import random

class BS:
    def __init__(self, loc, BStype):
        self.location= loc
        self.type = BStype
        
def distanceloc(loc1, loc2):
    return math.sqrt(pow(loc1[0]-loc2[0], 2) + pow(loc1[1]-loc2[1], 2));
def distance (BS1, BS2):
    loc1 = BS1.location
    loc2 = BS2.location
    return distanceloc(loc1, loc2)

def calculatecapacity(UE, BS, N_connected, variables, dofading=True):
    B = variables['B']
    No = variables['No']
    Pt = variables['P']
    K = variables['K_PL']
    alpha = variables['alpha']
    dist = distanceloc(UE.location, BS.location)
    Pr = Pt * K * pow(dist, -alpha)
    
    gamma_bar = Pr/B/No
    if dofading:
        gamma_bar = random.expovariate(1./gamma_bar) #for now, assume uncorrelated fading even when UEs on top of each other
    return B/N_connected * math.log(1 + gamma_bar, 2)
    
    
    