# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:14:04 2015

@author: Nikhil


components:
    Geometry of the network (remains constant) -  UEs + BSs

    variables that update each time step
        which BS's transmitting (1 or 0 for each BS)
        fading
    variables that don't update
        path loss, network parameters
    
    each "agent"
        has a location
        has a vector of past actions (which BS it connected to)
        has a vector of past rewards
        has a function that takes in past actions/rewards, time, some variables --> an action for next step
    
Gameplay:
    determine variables that update (either random or some algorithm)
    ask each agent for its play
    calculate rewards for each agent
        send this back to the agent
        store this somewhere
        
    continue with probability 1-beta. 
"""
