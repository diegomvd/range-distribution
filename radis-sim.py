# -*- coding: utf-8 -*-

import glob, os, sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
import seaborn as sns
from collections import OrderedDict

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

        # normal distribution for environment suitability
        def gaussian(x,A,mean,std):
            return A*exp(0.5*(x-mean)*(x-mean)/std)

        # returns list of the size of the individualsList with 0 when species
        # i doesn't coocur with species j and 1 when it does. coocurrence is
        # not symmetric as species can have different interaction radius
        def getCoOcurrence(i, radius, distanceMatrix, preyPopulationList):
            coOcurrence=np.zeros(len(preyPopulationList))
            for j in range(len(preyPopulationList)):
                if distanceMatrix(i,j)<=radius:
                    coOcurrence[j]=1
            return coOcurrence

        # given a species coorcurrence list and a list of prefered preys, the
        # function returns 1 if any of the coocurring species is in the prey list
        def getInteractions(coOcurrentIndividuals,listOfPreys):
            interaction=0
            i=0
            while interaction==0:
                if coOcurrentIndividuals[i]=1
                    if i.isin(listOfPreys):
                        interaction=1
                i=i+1
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
        environmentSurvival = gaussian(self.y,1,meanEnvironment,stdEnvironment)

        # probability of survival given the prefered diet and coocurring species
        coOcurrentIndividuals = getCoOcurrence(self.i, R, distanceMatrix, individualsList)
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
        # normal distribution for environment suitability
        def gaussian(x,A,mean,std):
            return A*exp(0.5*(x-mean)*(x-mean)/std)

        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd

        # probability of survival given the environment preferences
        environmentSurvival = gaussian(self.y,1,meanEnvironment,stdEnvironment)

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

def getDistance(xi,yi,xj,yj):
    return np.sqrt( ((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj)) )

###############################################################################
# end of classes and functions declaration
###############################################################################

# model Parameters
nPopulations=1000
nPredators=10
nPreys=10
kMean = 3
kStd = 3

param=0.5

# species initialization
predatorList=[]
for i in range(nPredators):
    mean = random(0,L)
    std = random(0,L/3)

    numOfPreys = random(kMean, kStd)
    listOfPreys=np.zeros(0)
    j=0
    while len(listOfPreys<numOfPreys):
        prey_id = random(0,nPreys)
        if ! np.isin(prey_id,listOfPreys):
            listOfPreys=np.append(listOfPreys,prey_id)

    radius = random(0,L/10)
    predatorList.append( Species(mean,std,listOfPreys,radius) )

preyList=[]
for i in range(nPreys):
    mean = random(0,L)
    std = random(0,L/3)

# population's initialization
predatorPopulationList=[]
for i in range(nPopulations):
    xpos = random(0,L)
    ypos = random(0,L)
    s = predatorList[random(nPredators)]
    predatorPopulationList.append( PredatorPopulation(xpos,ypos,s,i) )

preyPopulationList=[]
for i in range(nPopulations):
    xpos = random(0,L)
    ypos = random(0,L)
    s = preyList[random(nPreys)]
    preyPopulationList.append( PreyPopulation(xpos,ypos,s,i) )

distanceMatrix = np.zeros((nIndividuals,nIndividuals))

for i in range(nIndividuals):
    for j in range(nIndividuals):
        xi=individualsList[i].x
        yi=individualsList[i].y
        xj=individualsList[j].x
        yj=individualsList[j].y
        distanceMatrix[i,j] = getDistance(xi,yi,xj,yj)

for t in range(Time):

    for i in range(nPopulations):

        if(predatorPopulationList[i].survival_probability(param, predatorPopulationList, distanceMatrix)<random(0,1)):
            xpos = predatorPopulationList[i].x
            ypos = predatorPopulationList[i].y
            s = predatorList[random(nPredators)]
            predatorPopulationList[i]=PredatorPopulation(xpos,ypos,s,i)

        if(preyPopulationList[i].survival_probability()<random(0,1)):
            xpos = preyPopulationList[i].x
            ypos = preyPopulationList[i].y
            s = preyList[random(nPreys)]
            preyPopulationList[i]=PreyPopulation(xpos,ypos,s,i)  
