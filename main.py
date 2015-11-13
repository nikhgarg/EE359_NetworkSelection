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
import csv
import numpy as np
from agent import Agent, Agent_BasicLearning
import environment
from environment import BS
import random
from collections import Counter
import matplotlib.pyplot
import pylab
import time
input_file = 'experiments.csv'
csv_file = csv.DictReader(open(input_file, 'r'), delimiter=',', quotechar='"')

output_file = 'data.csv'

def getNetworkGeometry(CASE):
    BSs = []
    UElocs = []
    if CASE is 0:
        BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')]
        UElocs = [(0,0)]
    if CASE is 1:
        BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')]
        UElocs = [(0,0), (0,0)]
    return [BSs, UElocs]

def createAgents(CASE, UElocs, BSs):
    Agents= []
    actspace = range(0, len(BSs))
    for i in range(0,len(UElocs)):
        Agents.append(Agent_BasicLearning(UElocs[i], actspace))
    return Agents
  
def determineWhichBSTransmitting(BSs, variables):
    x = [0]*len(BSs)
    if COEXISTENCEPROTOCOL is 0: #each LTE-U node transmit with probability K. Each WiFi node transmits if no one around it is transmitting.
        K = variables["K_coexistence"]
        for i in range(0, len(BSs)):
            if BSs[i].type== 'LTE':
                if random.random() <= K : #true with probability K
                    x[i] = 1
    for i in range(0, len(BSs)):
        turnon = True
        for j in range(0, len(BSs)):
            if i==j:
                continue
            if environment.distance(BSs[i], BSs[j]) <= variables['ConnectionDistance'] and x[j] == 1:
                #don't turn this BS on.
                turnon = False;
        if turnon:
            x[i] = 1
    return x

def determineRewards(BSs, Agents, actions, variables):
    x = determineWhichBSTransmitting(BSs, variables);
    Ntx = Counter(actions)     #find number of UEs that chose each BS
    rewards = [0]*len(Agents)
    for i in range(0, len(Agents)):
        if x[actions[i]]  == 0: #BS it is connected to is not transmitting
            continue
        rewards[i] = environment.calculatecapacity(Agents[i], BSs[actions[i]], Ntx[actions[i]], variables)   
    return rewards

for configuration in csv_file:
    variables = configuration;
    for var in variables:
        if var != 'ExperimentName':
            variables[var]= float(variables[var])
        else:
            variables[var] = variables[var] + '_' + str(time.time());
    CASE = int(variables['CASE'])
    COEXISTENCEPROTOCOL = int(variables['COEXISTENCEPROTOCOL'])        
    NumExperiments = int(variables['NumRepeat']);
    AgentRewards = []
    for experiment_num in range(0, NumExperiments):
        [BSs, UElocs] = getNetworkGeometry(CASE);
        Agents = createAgents(CASE, UElocs, BSs)
        
        t = 0
        while (True):
        #    ask each agent for its play
            actions = []
            for agent in Agents:
                actions.append(agent.act(BSs, variables, t))
            rewards = determineRewards(BSs, Agents, actions, variables) #    calculate rewards for each agent
            for i in range(0, len(Agents)): #        send this back to the agent
                Agents[i].updatereward(rewards[i])    
            
        #    continue with probability 1-beta. (actually breaking at time T_cutoff for now)
            t = t+1
            if t >= variables['T_cutoff']:
                break
        for i in range(0, len(Agents)):
            if experiment_num is 0:
                AgentRewards.append(np.array(Agents[i].rewards))
            else:
                AgentRewards[i] = AgentRewards[i] + np.array(Agents[i].rewards)
    for i in range(0, len(Agents)):
        AgentRewards[i] = AgentRewards[i]/NumExperiments
    matplotlib.pyplot.scatter(range(0, int(variables['T_cutoff'])), AgentRewards[0])
    matplotlib.pyplot.xlabel('t')
    matplotlib.pyplot.ylabel('C_{avg}')
    matplotlib.pyplot.show()


    #output results:
    
    fieldnamesorder = ['ExperimentName', 'CASE', 'COEXISTENCEPROTOCOL', 'Agent_type', 'Agent_location']
    AgentPrintingDicts = []
    for i in range(0, len(Agents)):
        AgentPrintingDicts.append(variables.copy())
        AgentPrintingDicts[i]['Agent_location'] = Agents[i].location #NOTE Only valid for deterministic locations
        AgentPrintingDicts[i]['Agent_type'] = type(Agents[i]).__name__ #NOTE Only valid for deterministic locations/orderings

        
    for var in variables:
        if var not in fieldnamesorder:
            fieldnamesorder.append(var)
    for i in range(0, int(variables['T_cutoff'])):
        fieldnamesorder.append(str(i))
        for agent in range(0, len(Agents)):
            AgentPrintingDicts[agent][str(i)] = AgentRewards[agent][i]
    
    outputfilehandle = open(output_file, 'a')
    outputwriter = csv.DictWriter(outputfilehandle, fieldnames = fieldnamesorder, delimiter=',', quotechar='"', extrasaction='raise')
    outputwriter.writeheader();
    for i in range(0, len(Agents)):
        outputwriter.writerow(AgentPrintingDicts[i])
    outputfilehandle.close()

#for deterministic agents:
    #For each agent, aggregate/average their preferences over the experiments and report them
#for non-deterministic agents (later):
    #aggregate by type of agent or something, or combine all of them, or just do the middle?
    
print("done")
#def bar():  return 1
#mydct = {'foo': bar}
#mydct['foo']()
        