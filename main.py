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


from agent import Agent
from environment import BS
import random
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
    if COEXISTENCEPROTOCOL is 0:
        K = variables["K"]
        
    return [1]*len(BSs)

def determineRewards(actions, instvariables):
    return [1]*len(actions)
    

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
        