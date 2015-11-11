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
    

"""
CASE = 0
COEXISTENCEPROTOCOL = 0
K = .5
beta = .95
CONNECTIONDISTANCE = 3

from agent import Agent
import environment
from environment import BS
import random
from collections import Counter

def getNetworkGeometry(CASE):
    BSs = []
    UEs = []
    if CASE is 0:
        BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')]
        UEs= [(0,0)]
    return [BSs, UEs]

def createAgents(CASE, UEs, BSs):
    Agents= []
    actspace = range(0, len(BSs))
    for i in range(0,len(UEs)):
        Agents.append(Agent(UEs[i], actspace))
    return Agents

#dictionary of variables and functions that determine those variables at each time instant    
def createVariableDictionary():
    return {}
    
def determineWhichBSTransmitting(BSs, variables):
    x = [0]*len(BSs)
    if COEXISTENCEPROTOCOL is 0: #each LTE-U node transmit with probability K. Each WiFi node transmits if no one around it is transmitting.
        K = variables["K"]
        for i in range(0, len(BSs)):
            if BS.type == 'LTE':
                if random.random() <= K : #true with probability K
                    x[i] = 1
    for i in range(0, len(BSs)):
        turnon = True
        for j in range(0, len(BSs)):
            if i==j:
                continue
            if environment.distance(BSs[i], BSs[j]) <= CONNECTIONDISTANCE and x[j] == 1:
                #don't turn this BS on.
                turnon = False;
        if turnon:
            x[i] = 1
    return x

def determineRewards(BSs, UEs, actions, instvariables):
    x = determineWhichBSTransmitting(BSs, instvariables);
    Ntx = Counter(actions)     #find number of UEs that chose each BS
    rewards = [0]*len(UEs)
    for i in range(0, len(UEs)):
        if x[actions[i]]  == 0: #BS it is connected to is not transmitting
            continue
        rewards[i] = environment.calculatecapacity(UEs[i], BSs[actions[i]], Ntx, instvariables)   
    return rewards
    

[BSs, UEs] = getNetworkGeometry(CASE);
Agents = createAgents(CASE, UEs, BSs)


t = 0
while (True):
#    determine variables that update (either random or some algorithm)
    instvariables = createVariableDictionary()
#    ask each agent for its play
    actions = []
    for agent in Agents:
        actions.append(agent.act(t, instvariables))
    rewards = determineRewards(actions, instvariables) #    calculate rewards for each agent
    for i in range(0, len(Agents)): #        send this back to the agent
        Agents[i].updatereward(rewards[i])    
    
#    continue with probability 1-beta. 
    t = t+1    
    if random.random() > beta:
        break
    

#def bar():  return 1
#mydct = {'foo': bar}
#mydct['foo']()
        