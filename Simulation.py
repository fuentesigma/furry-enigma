#!/usr/bin/env python

__author__ = "Jesus Fuentes"
__version__ = "0.3"
__email__ = "jesus.fuentes@uni.lu"

# Parent modules
import Motion
# Basic Modules
import sys
import imageio
import numpy as np
from numpy.random import randn
from sklearn.neighbors import NearestNeighbors
# Vispy modules
import vispy
from vispy.scene import visuals
from vispy import app

# /////////////////////////////////////////////////////////////////////////////////

# Define a vispy-like canvas 
canvas = vispy.scene.SceneCanvas(title='Simulation', keys='interactive', show=True, bgcolor='black', size=(800,800))
# Settings for handling camera 
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 30
    
# Vispy objects
scatter = visuals.Markers(scaling=True, spherical=True, symbol='disc', alpha=1, edge_color='#0674EF', edge_width=10)
ABpolar = visuals.Arrow(arrow_type='triangle_30', antialias=True, connect="segments", width=1)
PCpolar = visuals.Arrow(arrow_type='triangle_30', antialias=True, connect="segments", width=1)

# Endowing canvas with objects
view.add(scatter)
view.add(ABpolar)
view.add(PCpolar)
axis = visuals.XYZAxis(parent=view.scene)

class Evolution:
    def __init__(self, radiovectors, ABP, PCP, i, sigma=[0.1, 0.8, 0.1], e=1, video=False, filename='video.mp4'):
        self.r = radiovectors
        self.p = ABP
        self.q = PCP
        self.s = sigma
        self.i = i
        self.t = 0
        self.e = e
        if video:
            self.filename = 'output/' + filename
            self.writer = imageio.get_writer(self.filename)
        
    def Gauss(self, X, step, noise=1e-5):
        self.noise = noise
        self.step = step
        a = np.random.normal(0, self.noise, size=(len(X),3))
        #b = np.random.normal(0, self.noise, size=(len(X),3))
        #a*np.sin(np.pi*self.step) + b*np.cos(np.pi*self.step)
        return a
    
    def simulation(self, t, D=1, T=1, kB=1):
        self.t = t
        # Coefficients
        c1 = D/(kB*T)
        c2 = np.sqrt(2*D)
        # Time evolution parameters
        dt = 0.05
        # Integration over time domain
        dr, dp, dq = Motion.evolution(self.r, self.p, self.q, sigma=self.s, e=self.e)
        self.r = self.r + c1*dt*dr + c2*self.Gauss(self.r, self.t*dt, noise=5e-4)
        self.p = self.p + c1*dt*dp + c2*self.Gauss(self.r, self.t*dt, noise=5e-4)
        self.q = self.q + c1*dt*dq + c2*self.Gauss(self.r, self.t*dt, noise=5e-4)
        self.p = Motion.unitary(self.p)
        self.q = Motion.unitary(self.q)
        return self.r, self.p, self.q

    def update(self, event):
        self.r_, self.p_, self.q_ = self.simulation(self.t)
        scatter.set_data(self.r_, face_color="#0674EF", size=100)
        ABpolar.set_data(np.stack((self.r_, self.r_ + self.p_), axis=1), color="#A31368")
        PCpolar.set_data(np.stack((self.r_, self.r_ + self.q_), axis=1), color="#60EC83")
        # Update progressbar
        progressbar(self.t+1, self.i, "Computing...")
        self.t += 1
    
    def movie(self, event):
        self.r_, self.p_, self.q_ = self.simulation(self.t)
        scatter.set_data(self.r_, face_color="#0674EF", size=100)
        ABpolar.set_data(np.stack((self.r_, self.r_ + self.p_), axis=1), color="#A31368")
        PCpolar.set_data(np.stack((self.r_, self.r_ + self.q_), axis=1), color="#60EC83")
        # Update movie
        self.writer.append_data(canvas.render(alpha=True))
        # Update progressbar
        progressbar(self.t+1, self.i, "Computing...")
        self.t += 1
        # Close writer
        if self.t+1 == self.i:
            self.writer.close()

    def show(self):
        canvas.show()

"""
    PROGRESS BAR
"""
def progressbar(j, count, prefix="", size=60, out=sys.stdout):
    x = int(size*j/(count-1))
    print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
          end='\r', file=out, flush=True)
        
"""
    EOF
"""