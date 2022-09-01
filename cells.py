#!/usr/bin/env python

__author__ = "J. Fuenters"
__version__ = "0.1"
__email__ = "jesus.fuentes@uni.lu"

# Basic modules
import numpy as np
from numpy.random import randn
from sklearn.neighbors import NearestNeighbors

"""
    MAIN ALGORITHM:
        R = RADIOVECTORS OF CELLS
        P = ABP VECTORS
        Q = PCP VECTORS
        BETA = FREE PARAMETER TO PLAY WITH
        SIGMA = CONTAINS THE STRENGHTS OF EACH POLARISATION
    
    ADJUST THE NUMBER OF NEIGHBOURS ACCORDING TO THE NUMBER OF CELLS
"""


def evolution(R, P, Q, beta=5, sigma=[1, 0, 0], NN=20):
    neighbours = NearestNeighbors(n_neighbors=NN, algorithm='kd_tree').fit(R)
    N = len(R)
    dr = np.zeros((N, 3))
    dp = np.zeros((N, 3))
    dq = np.zeros((N, 3))
    for i in range(0, N):
        # Obtain k-nns for the ith cell
        _, j = neighbours.kneighbors([R[i]])
        j = j[j!=i]
        # ABP and PCP vectors (ith cell and its neighbours)
        Pi = P[i]
        Pj = P[j]
        Qi = Q[i]
        Qj = Q[j]
        # Components of the radiovectors from the ith cell to its neighbours
        x = R[j,0] - R[i,0]
        y = R[j,1] - R[i,1]
        z = R[j,2] - R[i,2]
        # Unitaries
        d = np.sqrt(x**2 + y**2 + z**2)
        x = x/d
        y = y/d
        z = z/d
        # Projection of ri into Pi and Qi
        rPi = x*Pi[0] + y*Pi[1] + z*Pi[2]
        rQi = x*Qi[0] + y*Qi[1] + z*Qi[2]
        # Projection of ri into Pj and Qj
        rPj = x*Pj[:,0] + y*Pj[:,1] + z*Pj[:,2]
        rQj = x*Qj[:,0] + y*Qj[:,1] + z*Qj[:,2]
        # Inner product between Pi and Pj
        PiPj = Pi[0]*Pj[:,0] + Pi[1]*Pj[:,1] + Pi[2]*Pj[:,2]
        # Inner product between Qi and Qj
        QiQj = Qi[0]*Qj[:,0] + Qi[1]*Qj[:,1] + Qi[2]*Qj[:,2]
        # Inner product between Pi and Qj
        PiQj = Pi[0]*Qj[:,0] + Pi[1]*Qj[:,1] + Pi[2]*Qj[:,2]
        # Inner product between Qi and Pj
        QiPj = Qi[0]*Pj[:,0] + Qi[1]*Pj[:,1] + Qi[2]*Pj[:,2]
        # Component S1
        S1 = PiPj - rPi*rPj
        #S1 = ((y*Pi[2]-z*Pi[1])*(y*Pj[:,2]-z*Pj[:,1]) + (z*Pi[0]-x*Pi[2])*(z*Pj[:,0]-x*Pj[:,2]) + (x*Pi[1]-y*Pi[0])*(x*Pj[:,1]-y*Pj[:,0]))
        # Component S2 ---- Check after iterations
        S2 = PiPj * QiQj - PiQj * QiPj
        # Component S3 ---- Check after iterations
        S3 = QiQj  - rQi * rQj
        # Constitutive relation
        S = sigma[0]*S1 + sigma[1]*S2 + sigma[2]*S3
        # ----------------------------------------------
        # Differential - cells' positions: dV/dr
        # ----------------------------------------------
        # Gamma (residual)
        A = sigma[0]*rPi*rPj + sigma[2]*rQi*rQj
        g = np.exp(-d*(beta-1)/beta) - S/beta + 2/d*A
        # Components of the differential of r
        dr[i,0] = -sum(np.exp(-d/beta)*(g*x - sigma[0]/d*(rPi*Pj[:,0] + rPj*Pi[0]) - sigma[2]/d*(rQi*Qj[:,0] + rQj*Qi[0])))
        dr[i,1] = -sum(np.exp(-d/beta)*(g*y - sigma[0]/d*(rPi*Pj[:,1] + rPj*Pi[1]) - sigma[2]/d*(rQi*Qj[:,1] + rQj*Qi[1])))
        dr[i,2] = -sum(np.exp(-d/beta)*(g*z - sigma[0]/d*(rPi*Pj[:,2] + rPj*Pi[2]) - sigma[2]/d*(rQi*Qj[:,2] + rQj*Qi[2])))
        # ----------------------------------------------
        # Differential - ABP vectors: dV/dp
        # ----------------------------------------------
        dp[i,0] = -sum(np.exp(-d/beta)*(sigma[0]*(S1*Pi[0] - Pj[:,0] + rPj*x) + sigma[1]*(S2*Pi[0] - QiQj*Pj[:,0] + QiPj*Qj[:,0]))) 
        dp[i,1] = -sum(np.exp(-d/beta)*(sigma[0]*(S1*Pi[1] - Pj[:,1] + rPj*y) + sigma[1]*(S2*Pi[1] - QiQj*Pj[:,1] + QiPj*Qj[:,1])))
        dp[i,2] = -sum(np.exp(-d/beta)*(sigma[0]*(S1*Pi[2] - Pj[:,2] + rPj*z) + sigma[1]*(S2*Pi[2] - QiQj*Pj[:,2] + QiPj*Qj[:,2]))) 
        # ----------------------------------------------
        # Differential - PCP vectors: dV/dq
        # ----------------------------------------------
        dq[i,0] = -sum(np.exp(-d/beta)*(sigma[1]*(S2*Qi[0] - PiPj*Qj[:,0] + PiQj*Pj[:,0]) + sigma[2]*(S3*Qi[0] - Qj[:,0] + rQj*x)))
        dq[i,1] = -sum(np.exp(-d/beta)*(sigma[1]*(S2*Qi[1] - PiPj*Qj[:,1] + PiQj*Pj[:,1]) + sigma[2]*(S3*Qi[1] - Qj[:,1] + rQj*y)))
        dq[i,2] = -sum(np.exp(-d/beta)*(sigma[1]*(S2*Qi[2] - PiPj*Qj[:,2] + PiQj*Pj[:,2]) + sigma[2]*(S3*Qi[2] - Qj[:,2] + rQj*z)))
    return dr, dp, dq

"""
    ROUTINE FOR COMPUTING UNITARIES
"""
def unitary(Z):
    r = np.zeros(Z.shape)
    l = np.linalg.norm(Z, axis=1)[:,None]
    for i in range(len(Z)):
        if l[i] == 0.0:
            r[i] = (Z[i]*0.0)
        else:
            r[i] = Z[i] / l[i]
    return r

"""
    EOF
"""