#!/usr/bin/env python

__author__ = "Jesus Fuentes"
__version__ = "0.3"
__email__ = "jesus.fuentes@uni.lu"

# IMPORT MODULES
import sys
import Motion
import Simulation
import argparse
import numpy as np
import vispy
from vispy import app

# -----------------------------------------------------
# COMMAND LINE 
parser = argparse.ArgumentParser()
try:
    parser.add_argument('-n', type=int)
    parser.add_argument('-i', type=int)
    parser.add_argument('-s', type=float, nargs='+')
    parser.add_argument('-v', type=float)
    args = parser.parse_args()
except:
    print("USE: python controlpanel.py -n [CELLS] -i [ITERATIONS] -s [s1 s2 s3]")
    
val = args._get_kwargs()
_,ITERS = val[0]
_,NCELL = val[1]
_,SIGMA = val[2]

if sum(SIGMA) != 1.0:
    print("ERROR: s1 + s2 + s3 must sum 1.0")
    sys.exit(2)

# LOAD INITIAL CONDITIONS - SPHERE
R = Motion.rsphere(Motion.sphere(r=10, N=NCELL))
P = Motion.unitary(np.random.randn(len(R),3))
Q = Motion.unitary(np.random.randn(len(R),3))

# LAUNCH SIMULATION
s = Simulation.Evolution(R, P, Q, ITERS, sigma=SIGMA, video=False)
time = app.Timer()
# If <VIDEO = TRUE> then use time.connect(s.movie)
time.connect(s.update)
time.start(interval=0, iterations=ITERS)

if __name__ == '__main__':
    try:
        app.run()
    except RuntimeError:
        time.events.stop.connect(lambda x: app.quit())
