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
from agent import *
import environment
from environment import BS
import random
from collections import Counter
import matplotlib.pyplot
import pylab
import time
from scipy import stats
import analysis_helper
import math
input_file = 'experiments_k.csv'
csv_file = csv.DictReader(open(input_file, 'r'), delimiter=',', quotechar='"')
configs = []

for configuration in csv_file:
    configs.append(configuration)


output_file = 'data_k3.csv'

PLOTNEQ = 1

def getNetworkGeometry(CASE):
    BSs = []
    UElocs = []
    if CASE is 0:
        UElocs = [(0,0)]
    if CASE is 1:
        BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')]
        UElocs = [(0,0), (0,0)]
    return [BSs, UElocs]
    
AgentNameDictionary = {'Agent' : Agent, 'BasicLearning' : Agent_BasicLearning, 'BasicProportionalLearning': Agent_BasicProportionalLearning, 'StubbornLTE' : Agent_StubbornLTE, 'StubbornWiFi' : Agent_StubbornWiFi, 'FictitiousPlay' : Agent_FictitiousPlay, 'NaiveBayes': Agent_NaiveBayes, 'FictitiousProportionalPlay' : Agent_FictitiousProportionalPlay}

def createAgents(variables):
    Agents= []
    BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')] #hardcode for now fine.
    actspace = range(0, len(BSs))
    for i in range(0,variables['NumAgents']):
        xlocstring = 'Agent'+str(i)+'X'
        ylocstring = 'Agent'+str(i)+'Y'
        AgentTypeString = 'Agent'+str(i)+'Type'
        UEloc = [0,0]
        if variables['AgentLocs'] == 'Fixed' and xlocstring in list(variables.keys()):
            UEloc[0] = variables[xlocstring];
            UEloc[1] = variables[ylocstring]
        else:
            UEloc[0] = 2*random.random() - 1
            UEloc[1] = 2*random.random() - 1
        Ag = None
        if AgentTypeString in list(variables.keys()):
            Ag = AgentNameDictionary[variables[AgentTypeString]](UEloc, actspace, i)
        else:
            Ag = random.choice(list(AgentNameDictionary.values()))(UEloc, actspace, i)      
        Agents.append(Ag)
    return [Agents, BSs]
  
def determineWhichBSTransmitting(BSs, variables, t=-1, t_cutoff=-1):
    x = [0]*len(BSs)
    if COEXISTENCEPROTOCOL is 0: #each LTE-U node transmit with probability K. Each WiFi node transmits if no one around it is transmitting.
        K = variables["K_coexistence"]
 #       K = float(t)/t_cutoff; ## TODO undo. Currently testing the adaptation.
        for i in range(0, len(BSs)):
            if BSs[i].type== 'LTE':
                if random.random() <= K : #true with probability K
                    x[i] = 1
    for i in range(0, len(BSs)):
        if BSs[i].type == 'LTE':
            continue;
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

def determineRewards(BSs, Agents, actions, variables, t=-1, t_cutoff=-1):
    x = determineWhichBSTransmitting(BSs, variables, t, t_cutoff);
    Ntx = Counter(actions)     #find number of UEs that chose each BS
    rewards = [0]*len(Agents)
    for i in range(0, len(Agents)):
        if x[actions[i]]  == 0: #BS it is connected to is not transmitting
            continue
        rewards[i] = environment.calculatecapacity(Agents[i], BSs[actions[i]], Ntx[actions[i]], variables, doFading = False)   
    return rewards



epsilon= .0001
Agents = None
BSs = None

logsumaverageRewards = []
Ks = np.arange(0,1, .01)
logsumaverageRewards_Correlated = np.zeros(len(Ks))+ 2*math.log(epsilon, 2)
logsumaverageRewards_CorrelatedConstrained = np.zeros(len(Ks))+ 2*math.log(epsilon, 2)
logsumaverageRewards_MixedStrategies= [np.zeros(len(Ks))+ 2*math.log(epsilon, 2), np.zeros(len(Ks))+ 2*math.log(epsilon, 2), np.zeros(len(Ks))+ 2*math.log(epsilon, 2), np.zeros(len(Ks))+ 2*math.log(epsilon, 2), np.zeros(len(Ks))+ 2*math.log(epsilon, 2) ] #4 pure strategies, 1 mixed strategy
logsumaverageRewards_MixedMax = np.zeros(len(Ks))+ 2*math.log(epsilon, 2)
ConfigurationNames = []
index = -1
for Kcoex in Ks:
    index = index + 1
    AgentRewards_ALL = []
    AgentActions_ALL = []
    configindex = -1
    for configuration in configs:
        configindex = configindex + 1
        print(index, configindex)
        variables = configuration.copy();
        if index is 0:
            logsumaverageRewards.append(np.zeros(len(Ks)) + 2*math.log(epsilon, 2))
            ConfigurationNames.append(variables['ExperimentName'])

        if variables['VALID'] == '0':
            continue
        categoricalvariables = ['ExperimentName', 'Agent1Type', 'Agent0Type', 'AgentLocs']
        for var in configuration:
            if len(variables[var])>0 and (var not in categoricalvariables):
                variables[var]= float(variables[var])
            elif len(variables[var]) ==0:
                del variables[var]
            variables['ExperimentTime'] = str(time.time())
        COEXISTENCEPROTOCOL = int(variables['COEXISTENCEPROTOCOL'])        
        NumExperiments = int(variables['NumRepeat']);
        variables['NumAgents'] = int(variables['NumAgents']);
        variables['K_coexistence'] = Kcoex    
        AgentRewards = []
        AgentActions = []
        [Agents, BSs] = createAgents(variables) #copied here as well to calc mixed strategies when constant location
        for experiment_num in range(0, NumExperiments):
            [Agents, BSs] = createAgents(variables) #copied here as well to calc mixed strategies when constant location
            t = 0
            while (True):
            #    ask each agent for its play
                actions = []
                for agent in Agents:
                    actions.append(agent.act(BSs, variables, Agents, t))
                rewards = determineRewards(BSs, Agents, actions, variables, t, variables['T_cutoff']) #    calculate rewards for each agent
                for i in range(0, len(Agents)): #        send this back to the agent
                    Agents[i].updatereward(rewards[i], Agents)    
                
            #    continue with probability 1-beta. (actually breaking at time T_cutoff for now)
                t = t+1
                if t >= variables['T_cutoff']:
                    break
            for i in range(0, len(Agents)):
                if experiment_num is 0:
                    AgentRewards.append(np.array(Agents[i].rewards))
                    AgentActions.append(np.array(Agents[i].actions))
                else:
                    AgentRewards[i] = AgentRewards[i] + np.array(Agents[i].rewards)
                    AgentActions[i] = AgentActions[i] + np.array(Agents[i].actions)
        
        for i in range(0, len(Agents)):
            AgentRewards[i] = AgentRewards[i]/float(NumExperiments)
            AgentActions[i] = AgentActions[i]/float(NumExperiments)
            
        logsumaverageRewards[configindex][index] = math.log(epsilon +np.mean(AgentRewards[0][-100:-1]), 2) + math.log(epsilon + np.mean(AgentRewards[1][-100:-1]), 2)
        if configindex is 0: 
            foundcorre = False
            AllMixedStrategyRewards = []
            CorrelatedEquilRewards = np.zeros(variables['NumAgents'])
            CorrelatedEquilRewards_constrained = np.zeros(variables['NumAgents'])
        
            C = np.zeros((2, 2))
            for i in [0, 1]:
                for j in [0, 1]:
                    C[i,j] = environment.calculatecapacity(Agents[i], BSs[j], 1, variables, doFading=False)
                
            stratsmixed = analysis_helper.findALLPossibleStrategies(C, variables["K_coexistence"])
            for j in range(0, len(stratsmixed)):
                u1 = epsilon + analysis_helper.calcUtilityFromStrategy(0, stratsmixed[j], variables["K_coexistence"], C)
                u2 = epsilon + analysis_helper.calcUtilityFromStrategy(1, stratsmixed[j], variables["K_coexistence"], C)
                logval = math.log(u1, 2) + math.log(u2, 2)
                logsumaverageRewards_MixedStrategies[j][index] = logval
                logsumaverageRewards_MixedMax[index] = max(logsumaverageRewards_MixedMax[index], logval)
                
            foundcorre, corstrat, corrrewardsloc = analysis_helper.calc_correlated_equil_without_constraint(C, variables["K_coexistence"])
            if foundcorre:
                CorrelatedEquilRewards = corrrewardsloc    
                logsumaverageRewards_Correlated[index] = math.log(epsilon +CorrelatedEquilRewards[0], 2) + math.log(epsilon + CorrelatedEquilRewards[1], 2)
            foundcorre, corstrat, corrrewardsloc = analysis_helper.calc_correlated_equil(C, variables["K_coexistence"])
            if foundcorre:
                CorrelatedEquilRewards_constrained = corrrewardsloc    
                logsumaverageRewards_CorrelatedConstrained[index] = math.log(epsilon +CorrelatedEquilRewards_constrained[0], 2) + math.log(epsilon + CorrelatedEquilRewards[1], 2)

matplotlib.pyplot.plot(Ks, logsumaverageRewards_Correlated, label = 'Central Planner')
matplotlib.pyplot.plot(Ks, logsumaverageRewards_MixedMax, label = 'Best Pure Strategy')
matplotlib.pyplot.plot(Ks, logsumaverageRewards_CorrelatedConstrained, label = 'Constrained Central Planner')
matplotlib.pyplot.plot(Ks, logsumaverageRewards_MixedStrategies[4], label = 'Mixed Strategy ')
for jj in range(0, len(ConfigurationNames)):
    matplotlib.pyplot.plot(Ks, logsumaverageRewards[jj], marker = 'x', label = 'Experimental: ' + ConfigurationNames[jj])
matplotlib.pyplot.xlabel('K_coexistence')
matplotlib.pyplot.ylabel('logsum rate')
#frame1 = matplotlib.pyplot.gca()
#frame1.axes.get_yaxis().set_ticks([])
matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
matplotlib.pyplot.ylim([-5, -.5])
matplotlib.pyplot.show()