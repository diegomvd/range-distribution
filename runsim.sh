#!/bin/bash

<<COMMENT1
	Script to run several simulations.

	Parameters are:

			1   2   3   4    5    6    7     8       9        10     11
			T   Nx  Ny  dl param  C  nPred  nPrey  predPop  preyPop  SD

COMMENT1

#run

python3 radis-sim.py 10000 100 100 10 0.0 0.2 150 100 20000 20000 1 &
python3 radis-sim.py 10000 100 100 10 0.25 0.2 150 100 20000 20000 1 &
python3 radis-sim.py 10000 100 100 10 0.5 0.2 150 100 20000 20000 1 &
python3 radis-sim.py 10000 100 100 10 0.75 0.2 150 100 20000 20000 1 &
python3 radis-sim.py 10000 100 100 10 1.0 0.2 150 100 20000 20000 1 &
