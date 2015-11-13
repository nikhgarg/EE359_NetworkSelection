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
import numpy as np

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
    
class Agent_StubbornLTE(Agent):
    def act(self, BSs, variables, t = -1):
        action = 0 #TODO change when go to more complex networks     
        self.actions.append(action)
        return action

class Agent_StubbornWiFi(Agent):
    def act(self, BSs, variables, t = -1):
        action = 1 #TODO change when go to more complex networks     
        self.actions.append(action)
        return action

class Agent_ReinforcementLearning(Agent):
    def act(self, BSs, variables, t = -1): ## TODO write this code
        action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action

#With probability p, go to the one that maximized so far. with prob 1-p, do the other one
class Agent_BasicLearning(Agent):
    def act(self, BSs, variables, t = -1): ## TODO write this code
        p = .8;
        avgOfEach = np.zeros(len(BSs))
        for i in range(0,len(BSs)):
            indices = [ind for ind, j in enumerate(self.actions) if j == i]
            avgOfEach[i] = np.Inf if len(indices)==0 else sum([self.rewards[j] for j in indices])/(float(len(indices)))
        BSMax = np.argmax(avgOfEach)
        if random.random() < p:
            action = BSMax
        else:
            action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action