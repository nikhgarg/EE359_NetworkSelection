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
input_file = 'experiments.csv'
csv_file = csv.DictReader(open(input_file, 'r'), delimiter=',', quotechar='"')

output_file = 'data.csv'

PLOTNEQ = 1
PLOTLOGSUMOVERTIME = 1

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
    for experiment_num in range(0, NumExperiments):
        if experiment_num % 1000 == 1:
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
            if experiment_num is 0:
                AgentRewards.append(np.array(Agents[i].rewards))
                AgentActions.append(np.array(Agents[i].actions))
            else:
                AgentRewards[i] = AgentRewards[i] + np.array(Agents[i].rewards)
                AgentActions[i] = AgentActions[i] + np.array(Agents[i].actions)
    foundcorre     = False
    if PLOTNEQ:          
        AllMixedStrategyRewards = []
        AllMixedStrategyActions = []

#        MixedStrategyRewards = np.zeros(variables['NumAgents'])
        CorrelatedEquilRewards = np.zeros(variables['NumAgents'])
#        MixedStrategyActions = np.zeros(variables['NumAgents'])
        CorrelatedEquilActions = np.zeros(variables['NumAgents'])
        C = np.zeros((2, 2))
        for i in [0, 1]:
            for j in [0, 1]:
                C[i,j] = environment.calculatecapacity(Agents[i], BSs[j], 1, variables, doFading=False)
            
        for strat in analysis_helper.findPossibleStrategies(C, variables["K_coexistence"]):
             AllMixedStrategyRewards.append([analysis_helper.calcUtilityFromStrategy(0, strat, variables["K_coexistence"], C), analysis_helper.calcUtilityFromStrategy(1, strat, variables["K_coexistence"], C)])
             AllMixedStrategyActions.append([strat[0][1], strat[1][1]])
            
 #       bestmixed = analysis_helper.findBestMixedStrategy(C, variables["K_coexistence"])
 #       MixedStrategyRewards = MixedStrategyRewards + [analysis_helper.calcUtilityFromStrategy(0, bestmixed, variables["K_coexistence"], C), analysis_helper.calcUtilityFromStrategy(1, bestmixed, variables["K_coexistence"], C)]
 #       MixedStrategyActions = MixedStrategyActions + [bestmixed[0][1], bestmixed[1][1]]
        
        foundcorre, corstrat, corrrewardsloc = analysis_helper.calc_correlated_equil(C, variables["K_coexistence"])
 #       print(foundcorre, corstrat, corrrewardsloc)
        if foundcorre:
            CorrelatedEquilRewards = CorrelatedEquilRewards + corrrewardsloc
            CorrelatedEquilActions[0] = CorrelatedEquilActions[0] + 1-corstrat       
            CorrelatedEquilActions[1] = CorrelatedEquilActions[1] + corstrat            
#        else: #did not find a correlated equilibrium, (ie dominant strategy exists)
 #           CorrelatedEquilRewards = CorrelatedEquilRewards + corrrewardsloc
  #          CorrelatedEquilActions = CorrelatedEquilActions + [corstrat[0][1], corstrat[1][1]]
 #           print(CorrelatedEquilRewards)
    for i in range(0, len(Agents)):
        AgentRewards[i] = AgentRewards[i]/float(NumExperiments)
        AgentActions[i] = AgentActions[i]/float(NumExperiments)
#        CorrelatedEquilActions[i] = CorrelatedEquilActions[i]/float(NumExperiments)  COMMENTED OUT SINCE ONLY CALCULATING THOSE THINGS ONCE
#        CorrelatedEquilRewards[i] = CorrelatedEquilRewards[i]/float(NumExperiments)
#        MixedStrategyActions[i] = MixedStrategyActions[i]/float(NumExperiments)
#        MixedStrategyRewards[i] = MixedStrategyRewards[i]/float(NumExperiments)

    #visualize stuff.
    if PLOTLOGSUMOVERTIME:
        AgentRewardsLogsum = [math.log(epsilon +np.mean(AgentRewards[0][t]), 2) + math.log(epsilon + np.mean(AgentRewards[1][t]), 2) for t in range(0, int(variables['T_cutoff']))]        
        
        matplotlib.pyplot.plot(range(0, int(variables['T_cutoff'])), AgentRewardsLogsum, 'b', label='Actual Reward')
        matplotlib.pyplot.xlabel('t')
        matplotlib.pyplot.ylabel('Logsum Rate')
        matplotlib.pyplot.title('Logsum Rate over Time')
        if PLOTNEQ:         
            maxreward = AllMixedStrategyRewards[0][i]
            AllMixedStrategyRewards_Logsum = []
            for ii in range(0, len(AllMixedStrategyRewards)):
                AllMixedStrategyRewards_Logsum.append(math.log(epsilon +np.mean(AllMixedStrategyRewards[ii][0]), 2) + math.log(epsilon + np.mean(AllMixedStrategyRewards[ii][1]), 2))
            matplotlib.pyplot.axhline(y=AllMixedStrategyRewards_Logsum[0], color = 'r', marker = 'x', label = 'Mixed Strategy Rewards') 
            for ii in range(1, len(AllMixedStrategyRewards_Logsum)):
                matplotlib.pyplot.axhline(y=AllMixedStrategyRewards_Logsum[ii], color = 'r', marker = 'x') 
                maxreward = max(maxreward, AllMixedStrategyRewards_Logsum[ii])
            if foundcorre:
                CorrelatedEquilRewards_logsum = math.log(epsilon +np.mean(CorrelatedEquilRewards[0]), 2) + math.log(epsilon + np.mean(CorrelatedEquilRewards[1]), 2)
                matplotlib.pyplot.axhline(y=CorrelatedEquilRewards_logsum, color = 'g', label = 'Correlated Equil. Reward') 
                maxreward = max(CorrelatedEquilRewards_logsum, maxreward)
        matplotlib.pyplot.ylim([-5, -.5 ])
        matplotlib.pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        matplotlib.pyplot.show()
        matplotlib.pyplot.figure()

    # Two subplots, unpack the axes array immediately
    rewardaxs = [None, None];
    freward, (rewardaxs[0], rewardaxs[1]) = matplotlib.pyplot.subplots(1, 2, sharey=True)
    matplotlib.pyplot.suptitle('Rate over Time')


    for i in range(0, len(Agents)):
        rewardaxs[i].plot(range(0, int(variables['T_cutoff'])), AgentRewards[i], 'b')
        rewardaxs[i].set_xlabel('t')
        rewardaxs[i].set_ylabel('R_{avg}')
        rewardaxs[i].set_title('Agent' + str(i) + '\n' + Agents[i].name)
        if PLOTNEQ:      
            maxreward = AllMixedStrategyRewards[0][i]
            rewardaxs[i].axhline(y=AllMixedStrategyRewards[0][i], color = 'r', marker = 'x', linestyle = '--', label = 'Mixed Strategy Rewards') 
            for ii in range(1, len(AllMixedStrategyRewards)):
                rewardaxs[i].axhline(y=AllMixedStrategyRewards[ii][i], color = 'r', linestyle = '--', marker = 'x') 
                maxreward = max(maxreward, AllMixedStrategyRewards[ii][i])
            if foundcorre:
                rewardaxs[i].axhline(y=CorrelatedEquilRewards[i], color = 'g', linestyle = '--', label = 'Correlated Equil. Reward') 
                maxreward = max(CorrelatedEquilRewards[i], maxreward)
        rewardaxs[i].set_ylim([-.1, 1 ])
        if i is 1:
            rewardaxs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #matplotlib.pyplot.show()
    matplotlib.pyplot.show()
#    matplotlib.pyplot.figure()
#    
    actionaxs = [None, None]
    faction, (actionaxs[0], actionaxs[1]) = matplotlib.pyplot.subplots(1, 2, sharey=True)
    matplotlib.pyplot.suptitle('Actions over Time')
    for i in range(0, len(Agents)):
        actionaxs[i].plot(range(0, int(variables['T_cutoff'])), AgentActions[i], 'b')
        actionaxs[i].set_xlabel('t')
        actionaxs[i].set_ylabel('Avg Action')
        actionaxs[i].set_title('Agent' + str(i) + '\n' + Agents[i].name)
        actionaxs[i].set_ylim([-.1, 1.1])
        
        if PLOTNEQ:      
            actionaxs[i].axhline(y=AllMixedStrategyActions[0][i], color = 'r', linestyle='--', marker = 'x', label = 'Mixed Strategy') 
            for ii in range(1, len(AllMixedStrategyActions)):
                actionaxs[i].axhline(y=AllMixedStrategyActions[ii][i], color = 'r', linestyle='--', marker = 'x') 
            if foundcorre:
                actionaxs[i].axhline(y=CorrelatedEquilActions[i], color = 'g', linestyle='--', label = 'Correlated Equilibrium Action') 
        if i is 1:     
            actionaxs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #matplotlib.pyplot.show()
        #matplotlib.pyplot.figure()
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

#for deterministic agents:
    #For each agent, aggregate/average their preferences over the experiments and report them
#for non-deterministic agents (later):
    #aggregate by type of agent or something, or combine all of them, or just do the middle?
    
print("done")
#def bar():  return 1
#mydct = {'foo': bar}
#mydct['foo']()
        