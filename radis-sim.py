# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
from numpy.random import default_rng
rng = default_rng()

class PredatorPopulation:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.y = ypos
        self.species = s
        self.id = i

    # returns list of the size of the individualsList with 0 when species
    # i doesn't coocur with species j and 1 when it does. coocurrence is
    # not symmetric as species can have different interaction radius
    def getCoOcurrence(self, preyPopulationList):
        coOcurrence=np.zeros(len(preyPopulationList))
        for ix,preyPopulation in enumerate(preyPopulationList):
            if preyPopulation.x == self.x and preyPopulation.y == self.y:
                coOcurrence[ix]=1
        return coOcurrence

    # given a species coorcurrence list and a list of prefered preys, the
    # function returns 1 if any of the coocurring species is in the prey list
    def getNormalizedPreyWeights(self, param, preyPopulationList):
        # first determine the possible interactions and how much the predator
        # benefits from them
        coOcurrentPreys = self.getCoOcurrence(preyPopulationList)
        dietPreferences=self.species.preys

        preyWeight=np.zeros(len(preyPopulationList))
        dietPreferencesPop = np.zeros(len(preyPopulationList))
        for i in range(len(dietPreferencesPop)):
            sid=preyPopulationList[i].species.id
            dietPreferencesPop[i]=dietPreferences[sid]

        for i in range(len(preyWeight)):
            preyWeight[i]=coOcurrentPreys[i]*(dietPreferencesPop[i] - (dietPreferencesPop[i]-0.5)*param)
        maxPotentialWeight=1-0.5*param
        preyWeightNormal=preyWeight/maxPotentialWeight

        return preyWeightNormal

    def getDietSurvival(self,param,preyPopulationList):
        # get the survival probablility
        preyWeightNormal = self.getNormalizedPreyWeights(param,preyPopulationList)
        dietSurvival = np.max(preyWeightNormal)
        return dietSurvival

    def survival_probability(self, param, preyPopulationList, dl):

        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd

        # probability of survival given the environment preferences
        environmentSuitability = (gaussianFunction((self.y+1)*dl,1,meanEnvironment,stdEnvironment)-gaussianFunction((self.y)*dl,1,meanEnvironment,stdEnvironment))*0.5

        dietSurvival = self.getDietSurvival(param,preyPopulationList)

        # composed probability where param controls the relative importance of the
        # environment versus the diet on survival
        #return dietSurvival*(1 - (1-param)*(1-environmentSuitability) )
        return dietSurvival*(1-param*(1-environmentSuitability))

class PreyPopulation:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.y = ypos
        self.species = s
        self.id = i

    def survival_probability(self,dl):

        meanEnvironment=self.species.ymean
        stdEnvironment=self.species.ystd

        # probability of survival given the environment preferences
        environmentSurvival = (gaussianFunction((self.y+1)*dl,1,meanEnvironment,stdEnvironment)-gaussianFunction((self.y)*dl,1,meanEnvironment,stdEnvironment))*0.5

        return environmentSurvival

class Predator:

    def __init__(self, mean, std, list, i):
        self.ymean = mean
        self.ystd = std
        self.preys = list
        self.id = i

class Prey:

    def __init__(self, mean, std, i):
        self.ymean = mean
        self.ystd = std
        self.id = i

# normal distribution for environment suitability
def gaussianFunction(x,A,mean,std):
    return A*np.exp(-0.5*(x-mean)*(x-mean)/std/std)

def getNeighbours(x,y,xCells,yCells):
    north=[x,y+1]
    east=[x+1,y]
    south=[x,y-1]
    west=[x-1,y]
    if y+1 > yCells-1:
        north=[x,y-1] # reflective border conditions
    if x+1 > xCells-1:
        east=[0,y] # periodic border conditions
    if y-1 < 0:
        south=[x,y+1] # reflective
    if x-1 < 0:
        west=[xCells-1,y] # periodic

    return [north, east, south, west]

###############################################################################
# end of classes and functions declaration
###############################################################################

# model Parameters
Time=np.int(sys.argv[1])
xCells=np.int(sys.argv[2])
yCells=np.int(sys.argv[3])
dl=np.float(sys.argv[4])
param=np.float(sys.argv[5])
connectance=np.float(sys.argv[6])
nPredators=np.int(sys.argv[7])
nPreys=np.int(sys.argv[8])
nPredatorPopulations=np.int(sys.argv[9])
nPreyPopulations=np.int(sys.argv[10])
maxstd=np.double(sys.argv[11])
ncells=xCells*yCells

filename_prey= "DATA_RD_PREY_T_"+str(Time)+"_Nx_"+str(xCells)+"_Ny_"+str(yCells)+"_dl_"+str(dl)+"_param_"+str(param)+"_C_"+str(connectance)+"_nPred_"+str(nPredators)+"_nPrey"+str(nPreys)+"_predPop_"+str(nPredatorPopulations)+"_preyPop_"+str(nPreyPopulations)+"_SD_"+str(maxstd)+".csv"
filename_predator= "DATA_RD_PREDATOR_T_"+str(Time)+"_Nx_"+str(xCells)+"_Ny_"+str(yCells)+"_dl_"+str(dl)+"_param_"+str(param)+"_C_"+str(connectance)+"_nPred_"+str(nPredators)+"_nPrey"+str(nPreys)+"_predPop_"+str(nPredatorPopulations)+"_preyPop_"+str(nPreyPopulations)+"_SD_"+str(maxstd)+".csv"

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
    mean = rng.uniform(0,yCells*dl)
    std = rng.uniform(0,yCells*dl)
    dietPreferences=interactionMatrix[i,:]
    predatorList.append( Predator(mean,std,dietPreferences,i) )

# prey list initialization
preyList=[]
for i in range(nPreys):
    mean = rng.uniform(0,yCells*dl)
    std = rng.uniform(0,yCells*dl)
    preyList.append( Prey(mean,std,i) )

# predator's population initialization
predatorPopulationList=[]
for i in range(nPredatorPopulations):
    xpos = rng.integers(0,xCells)
    ypos = rng.integers(0,yCells)
    s = predatorList[rng.integers(0,nPredators)]
    predatorPopulationList.append( PredatorPopulation(xpos,ypos,s,i) )

# prey's population initialization
preyPopulationList=[]
for i in range(nPreyPopulations):
    xpos = rng.integers(0,xCells)
    ypos = rng.integers(0,yCells)
    s = preyList[rng.integers(0,nPreys)]
    preyPopulationList.append( PreyPopulation(xpos,ypos,s,i) )

###############################################################################

# begin of simulation
t0= time.time()
for t in range(Time):

    for i in range(nPreyPopulations):

        if rng.uniform() < (1-preyPopulationList[i].survival_probability(dl)):
            xpos = preyPopulationList[i].x
            ypos = preyPopulationList[i].y
            neighbourList = getNeighbours(xpos,ypos,xCells,yCells)
            #making a list with all the species present in the neighbourhood
            candidateReplacementSpecies=[]
            # think of replacement by choosing first the random neighbour and then the species
            # by randomly choosing a population. this takes into account the density of a given species
            for preyPop in preyPopulationList:
                pos = [preyPop.x,preyPop.y]
                for neighbour in neighbourList:
                    if pos==neighbour or pos==[xpos,ypos]:
                        candidateReplacementSpecies.append(preyPop.species.id)
                        break
            sid = candidateReplacementSpecies[rng.integers(0,len(candidateReplacementSpecies))]
            s = preyList[sid]
            preyPopulationList[i]=PreyPopulation(xpos,ypos,s,i)

    for i in range(nPredatorPopulations):

        if rng.uniform() < (1 - predatorPopulationList[i].survival_probability(param, preyPopulationList, dl)):
            xpos = predatorPopulationList[i].x
            ypos = predatorPopulationList[i].y
            #making a list with all the species present in the neighbourhood
            candidateReplacementSpecies=[]
            for predatorPop in predatorPopulationList:
                pos = [predatorPop.x,predatorPop.y]
                for neighbour in neighbourList:
                    if pos==neighbour or pos==[xpos,ypos]:
                        candidateReplacementSpecies.append(predatorPop.species.id)
                        break
            sid = candidateReplacementSpecies[rng.integers(0,len(candidateReplacementSpecies))]
            s= predatorList[sid]
            predatorPopulationList[i]=PredatorPopulation(xpos,ypos,s,i)

##############################################################################
# RESULTS ANALYSIS
##############################################################################

# returns the predators located in each cell
predatorPopByCell=np.zeros((xCells,yCells,nPredatorPopulations))
for predatorPopulation in predatorPopulationList:
    x=predatorPopulation.x
    y=predatorPopulation.y
    ix=predatorPopulation.id
    predatorPopByCell[x,y,ix]=1



# returns the preys located in each cell
preyPopByCell=np.zeros((xCells,yCells,nPreyPopulations))
for preyPopulation in preyPopulationList:
    x=preyPopulation.x
    y=preyPopulation.y
    ix=preyPopulation.id
    preyPopByCell[x,y,ix]=1

# print("Predator and prey populations per cell")
# print(predatorPopByCell[0,0,:])
# print(preyPopByCell[0,0,:])
# print("\n")

# initialize a list of the interaction matrixes
interactionMatrixList=np.zeros((ncells,nPredators,nPreys))

# store the interaction matrix for each cell
for x in range(xCells):
    for y in range(yCells):
        interactionMatrix=np.zeros(0)
        for ix,predatorPresence in enumerate(predatorPopByCell[x,y,:]):
            if predatorPresence == 1:
                cellid = x+xCells*y
                normalizedPreyWeights = predatorPopulationList[ix].getNormalizedPreyWeights(param,preyPopulationList)
                # print("Predator Population "+ str(ix) +" is of species")
                # print(predatorPopulationList[ix].species.id)
                # print("Normalized prey weights for the predator population "+str(ix))
                # print(normalizedPreyWeights)
                if np.any(normalizedPreyWeights>0):
                    normalizedPreyWeights/=np.max(normalizedPreyWeights)
                populationInteractionVector=normalizedPreyWeights
                populationInteractionVector[populationInteractionVector<1]=0
                # print("Interaction Vector for the predator population "+ str(ix))
                # print(populationInteractionVector)

                for jx,populationInteraction in enumerate(populationInteractionVector):
                    if populationInteraction == 1:
                        prey_sid=preyPopulationList[jx].species.id
                        predator_sid = predatorPopulationList[ix].species.id
                        interactionMatrixList[cellid,predator_sid,prey_sid]=1


# print("\n")
# print(interactionMatrixList[0,:,:])
# print("\n")

#global interaction matrix
globalInteractionMatrix = np.zeros((nPredators,nPreys))
for ix in range(ncells):
    for jx in range(nPredators):
        for kx in range(nPreys):
            if interactionMatrixList[ix,jx,kx]==1:
                globalInteractionMatrix[jx,kx]=1

# number of preys by predator
predatorDiet = np.transpose([np.sum(globalInteractionMatrix,axis=1)])
# print(predatorDiet)

# number of predators by prey
preyDiet = np.transpose([np.sum(globalInteractionMatrix,axis=0)])
# print(preyDiet)

# calculate the fraction of the total area occupied by each species
predatorOccupancyMatrix=np.zeros((ncells,nPredators))
for x in range(xCells):
    for y in range(yCells):
        for ix,predatorPresence in enumerate(predatorPopByCell[x,y,:]):
            if predatorPresence == 1:
                cellid = x+xCells*y
                sid = predatorPopulationList[ix].species.id
                predatorOccupancyMatrix[cellid,sid]=1

preyOccupancyMatrix=np.zeros((ncells,nPreys))
for x in range(xCells):
    for y in range(yCells):
        for ix,preyPresence in enumerate(preyPopByCell[x,y,:]):
            if preyPresence == 1:
                cellid = x+xCells*y
                sid = preyPopulationList[ix].species.id
                preyOccupancyMatrix[cellid,sid]=1

preyOccupancy = np.transpose([np.sum(preyOccupancyMatrix,axis=0)])
predatorOccupancy = np.transpose([np.sum(predatorOccupancyMatrix,axis=0)])

# print(predatorOccupancy)
# print(preyOccupancy)

saveArrayPredator=np.concatenate( (predatorDiet ,predatorOccupancy) , axis=1)
saveArrayPrey=np.concatenate((preyDiet,preyOccupancy),axis=1)

np.savetxt(filename_prey,saveArrayPrey)
np.savetxt(filename_predator,saveArrayPredator)

t1 = time.time()
print("Time elapsed: ", t1 - t0)
