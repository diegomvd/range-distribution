# -*- coding: utf-8 -*-

import glob, os, sys

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
        def getInteractions(coOcurrentIndividuals,listOfPreys):
            interaction=0
            for i in range(len(coOcurrentIndividuals)):
                if coOcurrentIndividuals[i]==1:
                    interaction=1
                    break
            return interaction        

        # given the realized trophic interactions this function returns the
        # probability that a species will survive based solely on diet
        def getDietSurvival(trophicInteractions):
            return trophicInteractions

        # species characteristics
        listOfPreys=self.species.preys
        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd
        R=self.species.radius

        # probability of survival given the environment preferences
        environmentSurvival = gaussianFunction(self.y,1,meanEnvironment,stdEnvironment)

        # probability of survival given the prefered diet and coocurring species
        coOcurrentIndividuals = getCoOcurrence(self.id, R, distanceMatrix, preyPopulationList)
        trophicInteractions = getInteractions(coOcurrentIndividuals, listOfPreys)
        dietSurvival = getDietSurvival(trophicInteractions)

        # composed probability where param controls the relative importance of the
        # environment versus the diet on survival
        return param * environmentSurvival + (1-param) * dietSurvival

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

    def __init__(self, mean, std, list, R):
        self.ymean = mean
        self.ystd = std
        self.preys = list
        self.radius = R

class Prey:

    def __init__(self, mean, std):
        self.ymean = mean
        self.ystd = std

# normal distribution for environment suitability
def gaussianFunction(x,A,mean,std):
    return A*np.exp(-0.5*(x-mean)*(x-mean)/std/std)

###############################################################################
# end of classes and functions declaration
###############################################################################

# model Parameters
nPredatorPopulations=1000
nPreyPopulations=1000
nPredators=10
nPreys=10
connectance=0.2
Time=1
Lx=100
Ly=100
param=0.5

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
    listOfPreys=np.where(interactionMatrix[i,:]==1)
    radius = rng.uniform(0,std/10)
    predatorList.append( Predator(mean,std,listOfPreys,radius) )

# prey list initialization
preyList=[]
for i in range(nPreys):
    mean = rng.uniform(0,Ly)
    std = rng.uniform(0,Ly/3)
    preyList.append( Prey(mean,std) )

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

# begin of simulation
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
