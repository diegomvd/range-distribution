# -*- coding: utf-8 -*-

import glob, os, sys
import time

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

import numpy as np
from numpy.random import default_rng
rng = default_rng()

class PredatorPopulation:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.y = ypos
        self.species = s
        self.id = i

    def survival_probability(self, param, preyPopulationList, distanceMatrix):

        ##################################################################
        # declaration of helper functions
        ##################################################################

        # returns list of the size of the individualsList with 0 when species
        # i doesn't coocur with species j and 1 when it does. coocurrence is
        # not symmetric as species can have different interaction radius
        def getCoOcurrence(i, radius, distanceMatrix, preyPopulationList):
            coOcurrence=np.zeros(len(preyPopulationList))
            for j in range(len(preyPopulationList)):
                if distanceMatrix[i,j]<=radius:
                    coOcurrence[j]=1
            return coOcurrence

        # given a species coorcurrence list and a list of prefered preys, the
        # function returns 1 if any of the coocurring species is in the prey list
        def getDietSurvival(coOcurrentPreys, dietPreferences, param, preyPopulationList):
            # first determine the possible interactions and how much the predator
            # benefits from them
            preyProbability=np.zeros(len(coOcurrentPreys))
            dietPreferencesPop = np.zeros(len(coOcurrentPreys))
            for i in range(len(dietPreferencesPop)):
                sid=preyPopulationList[i].species.id
                dietPreferencesPop[i]=dietPreferences[sid]

            for i in range(len(preyProbability)):
                preyProbability[i]=coOcurrentPreys[i]*(dietPreferencesPop[i] - (dietPreferencesPop[i]-0.5)*param)
            maxPotentialProb=1-0.5*param
            preyProbability=preyProbability/maxPotentialProb

            # get the survival probablility
            dietSurvival = np.max(preyProbability)

            return dietSurvival

        # species characteristics
        dietPreferences=self.species.preys
        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd
        R=self.species.radius

        # probability of survival given the environment preferences
        environmentSuitability = gaussianFunction(self.y,1,meanEnvironment,stdEnvironment)-gaussianFunction(meanEnvironment,1,meanEnvironment,stdEnvironment)

        # probability of survival given the prefered diet and coocurring species
        coOcurrentPreys = getCoOcurrence(self.id, R, distanceMatrix, preyPopulationList)
        dietSurvival = getDietSurvival(coOcurrentPreys, dietPreferences, param, preyPopulationList)

        # composed probability where param controls the relative importance of the
        # environment versus the diet on survival
        return dietSurvival*(1 + (1-param)*environmentSuitability )

class PreyPopulation:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.y = ypos
        self.species = s
        self.id = i

    def survival_probability(self):

        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd

        # probability of survival given the environment preferences
        environmentSurvival = gaussianFunction(self.y,1,meanEnvironment,stdEnvironment)

        return environmentSurvival

class Predator:

    def __init__(self, mean, std, list, R, i):
        self.ymean = mean
        self.ystd = std
        self.preys = list
        self.radius = R
        self.id = i

class Prey:

    def __init__(self, mean, std, i):
        self.ymean = mean
        self.ystd = std
        self.id = i

# normal distribution for environment suitability
def gaussianFunction(x,A,mean,std):
    return A*np.exp(-0.5*(x-mean)*(x-mean)/std/std)

def getPopCell(x,y,ncells,L):
    dl=np.int(L*L/ncells)
    n = np.int(L/dl)

    cellx = np.floor(x/dl)
    celly = np.floor(y/dl)

    cellid = celly*n + cellx

    return cellid

###############################################################################
# end of classes and functions declaration
###############################################################################

# model Parameters
nPredatorPopulations=1000
nPreyPopulations=1000
nPredators=10
nPreys=10
connectance=0.2
Time=100
Lx=100
Ly=100
param=0.5
ncells=Lx*Ly

###############################################################################

# create interaction matrix. rows are predators, cols are preys, a 1 means
# interaction and 0 no interaction.
interactionMatrix=np.zeros((nPredators,nPreys))
for i in range(nPredators):
    for j in range(nPreys):
        if rng.uniform()<connectance:
            interactionMatrix[i,j]=1

# predator list initialization
predatorList=[]
for i in range(nPredators):
    mean = rng.uniform(0,Ly)
    std = rng.uniform(0,Ly/3)
    dietPreferences=interactionMatrix[:,i]
    radius = rng.uniform(0,std/10)
    predatorList.append( Predator(mean,std,dietPreferences,radius,i) )

# prey list initialization
preyList=[]
for i in range(nPreys):
    mean = rng.uniform(0,Ly)
    std = rng.uniform(0,Ly/3)
    preyList.append( Prey(mean,std,i) )

# predator's population initialization
predatorPopulationList=[]
for i in range(nPredatorPopulations):
    xpos = rng.uniform(0,Lx)
    ypos = rng.uniform(0,Ly)
    s = predatorList[rng.integers(0,nPredators)]
    predatorPopulationList.append( PredatorPopulation(xpos,ypos,s,i) )

# prey's population initialization
preyPopulationList=[]
for i in range(nPreyPopulations):
    xpos = rng.uniform(0,Lx)
    ypos = rng.uniform(0,Ly)
    s = preyList[rng.integers(0,nPreys)]
    preyPopulationList.append( PreyPopulation(xpos,ypos,s,i) )

# creating distance matrix to determine later coOcurrence. rows are predators
# and cols are preys
distanceMatrix = np.zeros((nPredatorPopulations,nPreyPopulations))
for i in range(nPredatorPopulations):
    for j in range(nPreyPopulations):
        xi=predatorPopulationList[i].x
        yi=predatorPopulationList[i].y
        xj=preyPopulationList[j].x
        yj=preyPopulationList[j].y
        distanceMatrix[i,j] = np.sqrt( ((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj)) )

###############################################################################

# begin of simulation
t0= time.time()
for t in range(Time):

    for i in range(nPreyPopulations):

        if rng.uniform() < (1-preyPopulationList[i].survival_probability()):
            xpos = preyPopulationList[i].x
            ypos = preyPopulationList[i].y
            s = preyList[rng.integers(0,nPreys)]
            preyPopulationList[i]=PreyPopulation(xpos,ypos,s,i)

    for i in range(nPredatorPopulations):

        if rng.uniform() < (1 - predatorPopulationList[i].survival_probability(param, preyPopulationList, distanceMatrix)):
            xpos = predatorPopulationList[i].x
            ypos = predatorPopulationList[i].y
            s = predatorList[rng.integers(0,nPredators)]
            predatorPopulationList[i]=PredatorPopulation(xpos,ypos,s,i)







t1 = time.time()
print("Time elapsed: ", t1 - t0)
