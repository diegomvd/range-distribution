# -*- coding: utf-8 -*-

"""

lattice model of spatial prey predator networks


"""


import sys
import time
import numpy as np
from numpy.random import default_rng
rng = default_rng()

class PredatorPopulation:

    def __init__(self, xpos, s, i):
        self.x = xpos
        self.species = s
        self.id = i

    # returns list of the size of the individualsList with 0 when species
    # i doesn't coocur with species j and 1 when it does. coocurrence is
    # not symmetric as species can have different interaction radius
    def getCoOcurrence(self, preyPopulationList):
        coOcurrence=np.zeros(len(preyPopulationList))
        for ix,preyPopulation in enumerate(preyPopulationList):
            if preyPopulation.x == self.x:
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

    def survival_probability(self, param, preyPopulationList):

        meanEnvironment=self.species.xmean
        stdEnvironment=self.species.xstd

        # probability of survival given the environment preferences
        environmentSuitability = (gaussianFunction((self.x+1),1,meanEnvironment,stdEnvironment)-gaussianFunction((self.x),1,meanEnvironment,stdEnvironment))*0.5

        dietSurvival = self.getDietSurvival(param,preyPopulationList)

        # composed probability where param controls the relative importance of the
        # environment versus the diet on survival
        #return dietSurvival*(1 - (1-param)*(1-environmentSuitability) )
        return dietSurvival*(1-param*(1-environmentSuitability))

    def change_position(self, xnew):
        self.x = xnew

class PreyPopulation:

    def __init__(self, xpos, ypos, s, i):
        self.x = xpos
        self.species = s
        self.id = i

    def change_position(self, xnew):
        self.x = xnew

    def survival_probability(self):

        meanEnvironment=self.species.xmean
        stdEnvironment=self.species.xstd

        # probability of survival given the environment preferences
        environmentSurvival = (gaussianFunction((self.x+1),1,meanEnvironment,stdEnvironment)-gaussianFunction((self.x),1,meanEnvironment,stdEnvironment))*0.5

        return environmentSurvival

class Predator:

    def __init__(self, mean, std, list, i):
        self.xmean = mean
        self.xstd = std
        self.preys = list
        self.id = i

class Prey:

    def __init__(self, mean, std, i):
        self.xmean = mean
        self.xstd = std
        self.id = i

# normal distribution for environment suitability
def gaussianFunction(x,A,mean,std):
    return A*np.exp(-0.5*(x-mean)*(x-mean)/std/std)

def getNeighbours(x,xCells):
    north=[x+1]
    south=[x-1]
    if x+1 > xCells-1:
        north=[x-1] # reflective border conditions
    if x-1 < 0:
        south=[x+1] # reflective border conditions
    return [north, south]

###############################################################################
# end of classes and functions declaration
###############################################################################

# model Parameters
Time=np.int(sys.argv[1])
xCells=np.int(sys.argv[2])
param=np.float(sys.argv[3])
connectance=np.float(sys.argv[4])
nPredators=np.int(sys.argv[5])
nPreys=np.int(sys.argv[6])
nPredatorPopulations=np.int(sys.argv[7])
nPreyPopulations=np.int(sys.argv[8])
maxstd=np.double(sys.argv[9])
ncells=xCells

filename_prey= "DATA_RD_PREY_T_"+str(Time)+"_Nx_"+str(xCells)+"_param_"+str(param)+"_C_"+str(connectance)+"_nPred_"+str(nPredators)+"_nPrey"+str(nPreys)+"_predPop_"+str(nPredatorPopulations)+"_preyPop_"+str(nPreyPopulations)+"_SD_"+str(maxstd)+".csv"
filename_predator= "DATA_RD_PREDATOR_T_"+str(Time)+"_Nx_"+str(xCells)+"_param_"+str(param)+"_C_"+str(connectance)+"_nPred_"+str(nPredators)+"_nPrey"+str(nPreys)+"_predPop_"+str(nPredatorPopulations)+"_preyPop_"+str(nPreyPopulations)+"_SD_"+str(maxstd)+".csv"

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
    mean = rng.uniform(0,xCells)
    std = xCells*rng.random()*maxstd
    dietPreferences=interactionMatrix[i,:]
    predatorList.append( Predator(mean,std,dietPreferences,i) )

# prey list initialization
preyList=[]
for i in range(nPreys):
    mean = rng.uniform(0,xCells)
    std = xCells*rng.random()*maxstd
    preyList.append( Prey(mean,std,i) )

# predator's population initialization
predatorPopulationList=[]
for i in range(nPredatorPopulations):
    xpos = rng.integers(0,xCells)
    s = predatorList[rng.integers(0,nPredators)]
    predatorPopulationList.append( PredatorPopulation(xpos,s,i) )

# prey's population initialization
preyPopulationList=[]
for i in range(nPreyPopulations):
    s = preyList[rng.integers(0,nPreys)]
    spatialDistribution = np.zeros(xCells)
    for x in range(xCells):
        spatialDistribution[x] = gaussian(x,1,s.mean,s.std)
    spatialDistribution = np.cumsum(spatialDistribution)/np.sum(spatialDistribution)
    rix = rng.random()
    xpos=0
    while spatialDistribution[xpos]<rix:
        xpos=xpos+1

    preyPopulationList.append( PreyPopulation(xpos,s,i) )

###############################################################################

# begin of simulation
t0 = time.time()
for t in range(Time):

    for i in range(nPredatorPopulations):

        if rng.uniform() < (1 - predatorPopulationList[i].survival_probability(param, preyPopulationList)):
            # there is a migration
            xpos = predatorPopulationList[i].x
            neighbourList = getNeighbours(xpos,xCells)
            destination = neighbourList[rng.integers(0,2)]
            predatorPopulationList[i].change_position(destination)


##############################################################################
# RESULTS ANALYSIS
##############################################################################

# returns the predators located in each cell
predatorPopByCell=np.zeros((xCells,nPredatorPopulations))
for predatorPopulation in predatorPopulationList:
    x=predatorPopulation.x
    ix=predatorPopulation.id
    predatorPopByCell[x,ix]=1

# returns the preys located in each cell
preyPopByCell=np.zeros((xCells,nPreyPopulations))
for preyPopulation in preyPopulationList:
    x=preyPopulation.x
    ix=preyPopulation.id
    preyPopByCell[x,ix]=1

# initialize a list of the interaction matrixes
interactionMatrixList=np.zeros((xCells,nPredators,nPreys))

# store the interaction matrix for each cell
for x in range(xCells):
    interactionMatrix=np.zeros(0)
    # iterate over predator populations in cell x
    for ix,predatorPresence in enumerate(predatorPopByCell[x,:]):
        # if predator is in cell x ...
        if predatorPresence == 1:
            # get the prey weights at the end of the simulation
            normalizedPreyWeights = predatorPopulationList[ix].getNormalizedPreyWeights(param,preyPopulationList)
            # if the predator is actually eating someone...
            if np.any(normalizedPreyWeights>0):
                # normalize the the prey weights (NB: maybe this is useless as i think it's already normalized)
                normalizedPreyWeights/=np.max(normalizedPreyWeights)

            # this is the realized interactions once param has an effect on diet breadth
            populationInteractionVector=normalizedPreyWeights
            # all the prey weights which are not maximum are assumed to be zero:
            # i.e. the predator chooses the prey he likes the most among those that he can get
            populationInteractionVector[populationInteractionVector<1]=0

            # now iterate over the interaction of populations to get the species
            # that are interacting
            for jx,populationInteraction in enumerate(populationInteractionVector):
                if populationInteraction == 1:
                    # the prey populations are listed in order in the preyWeights so we can use jx
                    prey_sid=preyPopulationList[jx].species.id
                    # the predatr population we currently looking at is in position ix of the list
                    predator_sid = predatorPopulationList[ix].species.id
                    # this is the interaction matrix at each cell
                    interactionMatrixList[x,predator_sid,prey_sid]=1

#global interaction matrix
globalInteractionMatrix = np.zeros((nPredators,nPreys))
for ix in range(xCells):
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
predatorOccupancyMatrix=np.zeros((xCells,nPredators))
for x in range(xCells):
    for ix,predatorPresence in enumerate(predatorPopByCell[x,:]):
        if predatorPresence == 1:
            sid = predatorPopulationList[ix].species.id
            predatorOccupancyMatrix[x,sid]=1

preyOccupancyMatrix=np.zeros((ncells,nPreys))
for x in range(xCells):
    for ix,preyPresence in enumerate(preyPopByCell[x,:]):
        if preyPresence == 1:
            sid = preyPopulationList[ix].species.id
            preyOccupancyMatrix[x,sid]=1

preyOccupancy = np.transpose([np.sum(preyOccupancyMatrix,axis=0)])
predatorOccupancy = np.transpose([np.sum(predatorOccupancyMatrix,axis=0)])

# save the environmental spread and the initial list size
stdPredatorArray=np.zeros(len(predatorList))
listpredatorArray=np.zeros(len(predatorList))
for ix,predator in enumerate(predatorList):
    stdPredatorArray[ix] = predator.std
    listPredatorArray[ix] = len(predator.prey)

stdPreyArray=np.zeros(len(preyList))
for ix,prey in enumerate(preyList):
    stdPreyArray[ix] = prey.std

paramPreyArray = np.ones(len(preyList))*param
paramPredatorArray = np.ones(len(predatorList))*param

saveArrayPredator=np.concatenate( (paramPredatorArray, stdPredatorArray, listPredatorArray, predatorDiet ,predatorOccupancy) , axis=1)
saveArrayPrey=np.concatenate( (paramPreyArray, stdPreyArray, preyDiet, preyOccupancy),axis=1)

np.savetxt(filename_prey,saveArrayPrey,delimiter=",",header="phi,spread,Npreferedprey,Nrealizedprey,occupancy")
np.savetxt(filename_predator,saveArrayPredator,delimiter=",",header="phi,spread,Npredators,occupancy")

t1 = time.time()
print("Time elapsed: ", t1 - t0)
