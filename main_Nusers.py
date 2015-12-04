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
input_file = 'experiments_N.csv'
csv_file = csv.DictReader(open(input_file, 'r'), delimiter=',', quotechar='"')

output_file = 'data.csv'

PLOTNEQ = 0
PLOTLOGSUMOVERTIME = 0

def getNetworkGeometry(CASE):
    BSs = []
    UElocs = []
    if CASE is 0:
        UElocs = [(0,0)]
    if CASE is 1:
        BSs = [BS((-1, 0), 'LTE'), BS((1, 0), 'WiFi')]
        UElocs = [(0,0), (0,0)]
    return [BSs, UElocs]
    
AgentNameDictionary = {'Agent' : Agent, 'BasicLearning' : Agent_BasicLearning, 'BasicProportionalLearning': Agent_BasicProportionalLearning, 'StubbornLTE' : Agent_StubbornLTE, 'StubbornWiFi' : Agent_StubbornWiFi, 'FictitiousPlay' : Agent_FictitiousPlay, 'NaiveBayes': Agent_NaiveBayes, 'FictitiousProportionalPlay' : Agent_FictitiousProportionalPlay, 'StubbornThenLearning':Agent_StubbornThenLearning, 'Stubborn': Agent_Stubborn}
randomchoices = ['Stubborn', 'BasicLearning', 'StubbornThenLearning']
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
            Ag = AgentNameDictionary[variables[AgentTypeString]](UEloc, actspace, i, name = variables[AgentTypeString])
        else:
            randkey = random.choice(randomchoices)
            Ag = AgentNameDictionary[randkey](UEloc, actspace, i, name = randkey)      
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
        rewards[i] = environment.calculatecapacity(Agents[i], BSs[actions[i]], Ntx[actions[i]], variables, doFading=False)   
    return rewards

epsilon= .0001
for configuration in csv_file:
    variables = configuration.copy();
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

    AgentRewards = []
    AgentActions = []
    AgentTypesRewards = [[]]*(len(randomchoices))
    AgentTypesActions = [[]]*(len(randomchoices))
    TypeCounts = np.zeros(len(randomchoices))
    
    for experiment_num in range(0, NumExperiments):
        if experiment_num % 100 == 1:
                print(experiment_num) 
        [Agents, BSs] = createAgents(variables)
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
            ag = Agents[i]
            typeind = randomchoices.index(ag.name)
            TypeCounts[typeind] = TypeCounts[typeind]  + 1
            if experiment_num is 0:
                AgentRewards.append(np.array(Agents[i].rewards))
                AgentActions.append(np.array(Agents[i].actions))
            else:
                AgentRewards[i] = AgentRewards[i] + np.array(Agents[i].rewards)
                AgentActions[i] = AgentActions[i] + np.array(Agents[i].actions)
            if len(AgentTypesRewards[typeind]) == 0:
                AgentTypesRewards[typeind] = ag.rewards
                AgentTypesActions[typeind] = ag.actions
            else:
                AgentTypesRewards[typeind] = np.add(AgentTypesRewards[typeind],ag.rewards)
                AgentTypesActions[typeind] = np.add(AgentTypesActions[typeind] ,ag.actions)

    for i in range(0, len(Agents)):
        AgentRewards[i] = AgentRewards[i]/float(NumExperiments)
        AgentActions[i] = AgentActions[i]/float(NumExperiments)

    #visualize stuff.
    if PLOTLOGSUMOVERTIME:
        AgentRewardsLogsum = [math.log(epsilon +np.mean(AgentRewards[0][t]), 2) + math.log(epsilon + np.mean(AgentRewards[1][t]), 2) for t in range(0, int(variables['T_cutoff']))]        
        
        matplotlib.pyplot.plot(range(0, int(variables['T_cutoff'])), AgentRewardsLogsum, 'b', label='Actual Reward')
        matplotlib.pyplot.xlabel('t')
        matplotlib.pyplot.ylabel('Logsum Rate')
        matplotlib.pyplot.title('Logsum Rate over Time')
        matplotlib.pyplot.ylim([-5, -.5 ])
        matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        matplotlib.pyplot.show()
#        matplotlib.pyplot.figure()
        
        
#aggregate by figure type -- plot rewards for each type of user
    print(TypeCounts)
    for typeind in range(0, len(randomchoices)):
        AgentTypesRewards[typeind] = AgentTypesRewards[typeind]/TypeCounts[typeind]
        AgentTypesActions[typeind] = AgentTypesActions[typeind]/TypeCounts[typeind]         

    for i in range(0, len(AgentTypesActions)):
        matplotlib.pyplot.plot(range(0, int(variables['T_cutoff'])), AgentTypesRewards[i], label=randomchoices[i])
    matplotlib.pyplot.xlabel('t')
    matplotlib.pyplot.ylabel('R_{avg}')
    matplotlib.pyplot.title('Rates over Time')
    matplotlib.pyplot.ylim([0, max([max(AgentTypesRewards[i]) for i in range(0, len(AgentTypesRewards))]) + .1])
    matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    matplotlib.pyplot.show()
            
    for i in range(0, len(AgentTypesActions)):
        matplotlib.pyplot.plot(range(0, int(variables['T_cutoff'])), AgentTypesActions[i], label = randomchoices[i])
    matplotlib.pyplot.xlabel('t')
    matplotlib.pyplot.ylabel('Avg Action')
    matplotlib.pyplot.title('Actions over Time')
    matplotlib.pyplot.ylim([-.1, 1.1])
    matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    matplotlib.pyplot.show()
    #output results:
    
    fieldnamesorder = ['ExperimentName', 'ExperimentTime', 'NumAgents', 'COEXISTENCEPROTOCOL', 'Agent_type', 'Agent_location']
    AgentPrintingDicts = []
    for i in range(0, len(Agents)):
        AgentPrintingDicts.append(variables.copy())
        AgentPrintingDicts[i]['Agent_location'] = Agents[i].location #NOTE Only valid for deterministic locations
        AgentPrintingDicts[i]['Agent_type'] = type(Agents[i]).__name__ #NOTE Only valid for deterministic locations/orderings

    NotToPrint = ['Agent0Type', 'Agent1Type', 'Agent0X', 'Agent1X', 'Agent0Y', 'Agent1Y', 'AgentLocs']
    for name in NotToPrint:
        if name in variables:
            del variables[name]
    for var in variables:
        if var not in fieldnamesorder:
            fieldnamesorder.append(var)
            
    for i in range(0, int(variables['T_cutoff'])):
        fieldnamesorder.append(str(i))
        for agent in range(0, len(Agents)):
            AgentPrintingDicts[agent][str(i)] = AgentRewards[agent][i]
    for i in range(0, int(variables['T_cutoff'])):
        fieldnamesorder.append("Action_" + str(i))
        for agent in range(0, len(Agents)):
            AgentPrintingDicts[agent]["Action_" + str(i)] = AgentActions[agent][i]    
    outputfilehandle = open(output_file, 'a', newline='')
    outputwriter = csv.DictWriter(outputfilehandle, fieldnames = fieldnamesorder, delimiter=',', quotechar='"', extrasaction='ignore')
   
    outputwriter.writeheader();
    for i in range(0, len(Agents)):
        outputwriter.writerow(AgentPrintingDicts[i])
    outputfilehandle.close()

    
print("done")
