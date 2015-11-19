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
    
    def act(self, BSs, variables, Agents=None, t = -1):
        action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action
        
    def updatereward(self, reward):
        self.rewards.append(reward)
    
class Agent_StubbornLTE(Agent):
    def act(self, BSs, variables,Agents=None, t = -1):
        action = 0 #TODO change when go to more complex networks     
        self.actions.append(action)
        return action

class Agent_StubbornWiFi(Agent):
    def act(self, BSs, variables, Agents=None, t = -1):
        action = 1 #TODO change when go to more complex networks     
        self.actions.append(action)
        return action

class Agent_ReinforcementLearning(Agent):
    def act(self, BSs, variables,Agents=None, t = -1): ## TODO write this code
        action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action

#With probability p, go to the one that maximized so far. with prob 1-p, do the other one
class Agent_BasicLearning(Agent):
    def act(self, BSs, variables,Agents=None, t = -1):
        p = 1-variables['p_explore'];
        if random.random() < p:
            avgOfEach = np.zeros(len(BSs))
            for i in range(0,len(BSs)):
                indices = [ind for ind, j in enumerate(self.actions) if j == i]
                avgOfEach[i] = np.Inf if len(indices)==0 else sum([self.rewards[j] for j in indices])/(float(len(indices)))
            action = np.argmax(avgOfEach)
        else:
            action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action
        
#With probability p, go to the one that maximized so far. with prob 1-p, do the other one
class Agent_BasicProportionalLearning(Agent):
    def act(self, BSs, variables,Agents=None, t = -1):
        pexplore = 1-variables['p_explore'];
        avgOfEach = np.zeros(len(BSs))
        if random.random() < pexplore:
            for i in range(0,len(BSs)):
                indices = [ind for ind, j in enumerate(self.actions) if (j == i and t - ind < 20)]
                avgOfEach[i] = 1e20 if len(indices)==0 else sum([self.rewards[j] for j in indices])/(float(len(indices)))
            sumavg = sum(avgOfEach)
            avgOfEach = [i/sumavg for i in avgOfEach]
            cdf = avgOfEach
            for i in range(1, len(cdf)):
                cdf[i] = cdf[i-1] + avgOfEach[i];
            val= random.random();
            action = 0;
            for i in range(0, len(cdf)):
                if cdf[i] > val:
                    action = i;
                    break;
        else:
            action = random.randint(0, len(BSs)-1) 
        self.actions.append(action)
        return action