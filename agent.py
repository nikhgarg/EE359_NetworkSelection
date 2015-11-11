# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:27:06 2015

@author: Nikhil

has a location
has an actionspace (of BSs it can connect to)
has a vector of past actions (which BS it connected to)
has a vector of past rewards
has a function that takes in past actions/rewards, time, some variables --> an action for next step
    
"""
import random

class Agent:
    def __init__(self, loc, actspace, name = ''):
        self.name = name
        self.actions = []
        self.rewards = []
        self.location= loc
        self.actionspace = actspace
    
    def act(self, BSs, variables, t = -1):
        action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action
        
    def updatereward(self, reward):
        self.rewards.append(reward)
    
    
    