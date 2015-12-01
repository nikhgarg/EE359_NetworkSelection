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
from sklearn.naive_bayes import BernoulliNB

class Agent:
    def __init__(self, loc, actspace, index, name = ''):
        self.name = name
        self.actions = []
        self.rewards = []
        self.location= loc
        self.actionspace = actspace
        self.index= index
    
    def act(self, BSs, variables, Agents=None, t = -1):
        action = random.randint(0, len(BSs)-1)      
        self.actions.append(action)
        return action
        
    def updatereward(self, reward, Agents = None):
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
        if random.random() < pexplore:
            avgOfEach = np.zeros(len(BSs))
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

#current class only works with 2 agents. Would explode exponentially with more.        
class Agent_FictitiousPlay(Agent):
    def __init__(self, loc, actspace, index, name = ''):
        super().__init__(loc, actspace, index)
        self.distribution = np.zeros((len(actspace), len(actspace))) #joint distribution (with probabilities)
        self.distribution_rewards = np.zeros((len(actspace), len(actspace))) #joint distribution (with rewards)

    def updatereward(self, reward, Agents):
        super().updatereward(reward)
        # update join probabilities (list of who did what)
        myact = self.actions[-1]
        otheract = Agents[1-self.index].actions[-1]
        self.distribution[myact, otheract] = self.distribution[myact, otheract] + 1
        self.distribution_rewards[myact, otheract] = self.distribution_rewards[myact, otheract] + reward

    def act(self, BSs, variables,Agents, t):
        pexplore = 1-variables['p_explore'];
        if random.random() < pexplore and t > 10: #exploit stage
            #calculate expected gain for each of your actions
                    # (each row is an action you can take, column is what they can take)
                    #Find probability of each of their actions.
            other_probabilities = [sum(self.distribution[:,i]) for i in range(0, len(BSs))]
            #For each of your actions, find avg from each of their actions times the probability of that action
            avgOfEach = np.zeros(len(BSs))
            for i in range(0, len(BSs)):
                summ = 0;
                for j in range(0, len(BSs)):
                    if self.distribution[i,j] > 0:
                        summ = summ + (self.distribution_rewards[i,j]/self.distribution[i,j])*other_probabilities[j]
                avgOfEach[i] = summ
            
            #choose action that maximizes expected
            action = avgOfEach.argmax()
        else: #explore stage
            action = random.randint(0, len(BSs)-1) 
        self.actions.append(action)
        return action
        
class Agent_FictitiousProportionalPlay(Agent):
    def __init__(self, loc, actspace, index, name = ''):
        super().__init__(loc, actspace, index)
        self.distribution = np.zeros((len(actspace), len(actspace))) #joint distribution (with probabilities)
        self.distribution_rewards = np.zeros((len(actspace), len(actspace))) #joint distribution (with rewards)

    def updatereward(self, reward, Agents):
        super().updatereward(reward)
        # update join probabilities (list of who did what)
        myact = self.actions[-1]
        otheract = Agents[1-self.index].actions[-1]
        self.distribution[myact, otheract] = self.distribution[myact, otheract] + 1
        self.distribution_rewards[myact, otheract] = self.distribution_rewards[myact, otheract] + reward

    def act(self, BSs, variables,Agents, t):
        pexplore = 1-variables['p_explore'];
        if random.random() < pexplore and t > 10: #exploit stage
            #calculate expected gain for each of your actions
                    # (each row is an action you can take, column is what they can take)
                    #Find probability of each of their actions.
            other_probabilities = [sum(self.distribution[:,i]) for i in range(0, len(BSs))]
            #For each of your actions, find avg from each of their actions times the probability of that action
            avgOfEach = np.zeros(len(BSs))
            for i in range(0, len(BSs)):
                summ = 0;
                for j in range(0, len(BSs)):
                    if self.distribution[i,j] > 0:
                        summ = summ + (self.distribution_rewards[i,j]/self.distribution[i,j])*other_probabilities[j]
                avgOfEach[i] = summ
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
        else: #explore stage
            action = random.randint(0, len(BSs)-1) 
        self.actions.append(action)
        return action
        
class Agent_NaiveBayes(Agent): #TODO. Currently only works with 1 other user. 
    def __init__(self, loc, actspace, index, name = ''):
        super().__init__(loc, actspace, index)
        self.distribution = np.zeros((len(actspace), len(actspace))) #joint distribution (with probabilities)
        self.distribution_rewards = np.zeros((len(actspace), len(actspace))) #joint distribution (with rewards)
        self.model= BernoulliNB();
        self.windowsize = 10; # Configurable
        self.predictions = np.zeros((self.windowsize, 1))
        
    def updatereward(self, reward, Agents):
        super().updatereward(reward)
        # update join probabilities (list of who did what)
        myact = self.actions[-1]
        otheract = Agents[1-self.index].actions[-1]
        self.distribution_rewards[myact, otheract] = self.distribution_rewards[myact, otheract] + reward
        self.distribution[myact, otheract] = self.distribution[myact, otheract] + 1

        #update model:
        OtherActions = Agents[1-self.index].actions;
        if len(OtherActions) > self.windowsize + 1:
            self.model.partial_fit(OtherActions[-self.windowsize-2:-2], [OtherActions[-1]], classes = self.actionspace)

    def act(self, BSs, variables,Agents, t): ## TODO write this code
        pexplore = 1-variables['p_explore'];
        if random.random() < pexplore and t > self.windowsize + 2: #exploit stage
            #predict what the other user will do
            others_predict = int(round(self.model.predict(Agents[1-self.index].actions[-self.windowsize-1:-1])[0]));
            np.insert(self.predictions, others_predict, len(self.predictions))            
            #find the action that maximizes assuming the the other does the predicted value
            avgOfEach = np.zeros(len(BSs))
            for i in range(0, len(BSs)):
                if self.distribution[i,others_predict] > 0:
                    avgOfEach[i] = self.distribution_rewards[i,others_predict]/self.distribution[i,others_predict]
            #choose action that maximizes expected
            action = avgOfEach.argmax()
        else: #explore stage
            action = random.randint(0, len(BSs)-1) 
        self.actions.append(action)
        return action