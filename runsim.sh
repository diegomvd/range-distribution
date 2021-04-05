#!/bin/bash

<<COMMENT1
	Script to run several simulations.

	Parameters are:

			1   2    3     4   5      6      7        8       9
			T   Nx  param  C  nPred  nPrey  predPop  preyPop  SD

COMMENT1

#run
python3 radis-sim.py 1000 1000 0.01 0.2 100 100 10000 10000 0.33&
python3 radis-sim.py 1000 1000 0.20 0.2 100 100 10000 10000 0.33&
python3 radis-sim.py 1000 1000 0.40 0.2 100 100 10000 10000 0.33&
python3 radis-sim.py 1000 1000 0.60 0.2 100 100 10000 10000 0.33&
python3 radis-sim.py 1000 1000 0.80 0.2 100 100 10000 10000 0.33&
python3 radis-sim.py 1000 1000 1.00 0.2 100 100 10000 10000 0.33&
