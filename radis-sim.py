# -*- coding: utf-8 -*-

import glob, os, sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
import seaborn as sns
from collections import OrderedDict

class Individuals:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.y = ypos
        self.species = s
        self.id = i

    def survival_probability(self, param, individualsList, distanceMatrix):

        def gaussian(x,A,mean,std):
            return A*exp(0.5*(x-mean)*(x-mean)/std)

        def getCoOcurrence(i, distanceMatrix, individualsList):
            coOcurrence=np.zeros(len(individualsList))
            for j in range(len(individualsList)):
                if distanceMatrix(i,j)<=self.species.radius:
                    coOcurrence[j]=1
            return coOcurrence

        def getInteractions(coOcurrentIndividuals,listOfPreys):
            interaction=0
            i=0
            while interaction==0:
                if coOcurrentIndividuals[i].isin(listOfPreys):
                    interaction=1
                i=i+1
            return interaction

        def getDietSurvival(trophicInteractions):
            return trophicInteractions

        listOfPreys=self.species.preys
        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd

        environmentSurvival = gaussian(self.y,1,meanEnvironment,stdEnvironment)

        coOcurrentIndividuals = getCoOcurrence(self.i, distanceMatrix, individualsList)
        trophicInteractions = getInteractions(coOcurrentIndividuals, listOfPreys)

        dietSurvival = getDietSurvival(trophicInteractions)

        return param * environmentSurvival + (1-param) * dietSurvival


class Species:

    def __init__(self, mean, std, list, R):
        self.ymean = xpos
        self.ystd = ypos
        self.preys = list
        self.radius = R



def getDistance(xi,yi,xj,yj): # define distance in periodic borders

###############################################################################

nIndividuals=1000
nSpecies=10
param=0.5

individualsList=[]

for i in range(nIndividuals):
    xpos = random()
    ypos = random()
    s = random()
    individualsList.append( Individuals(xpos,ypos,s,i) )

distanceMatrix = np.zeros((nIndividuals,nIndividuals))

for i in range(nIndividuals):
    for j in range(nIndividuals):
        xi=individualsList[i].x
        yi=individualsList[i].y
        xj=individualsList[j].x
        yj=individualsList[j].y
        distanceMatrix[i,j] = getDistance(xi,yi,xj,yj)

for t in range(Time):

    for i in range(nIndividuals):

        if(individualsList[i].survival_probability(param, individualsList, distanceMatrix)<random(0,1)):
            xpos = individualsList[i].x
            ypos = individualsList[i].y
            s = random()
            individualsList[i]=Individuals(xpos,ypos,s,i)
