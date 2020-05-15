"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Script to compute trajectories for the autonomous double-gyre flow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def vel(y,t):
    x1 = y[0]
    x2 = y[1]
    u = -np.pi * np.sin(np.pi * x1) * np.cos(np.pi * x2)
    v = np.pi * np.cos(np.pi * x1) * np.sin(np.pi * x2)
    return [u,v]


tau = 20 #integration time
dt_output = 0.1 #output time for trajectories
dt_int = 0.001 #integration time step
n_it = int(dt_output / dt_int) #number of integration steps

t = np.arange(0, tau+dt_output, dt_int)
x_range = np.linspace(0,2,202)[1:-1]
y_range = np.linspace(0,1,102)[1:-1]

#Initial conditions
X0, Y0 = np.meshgrid(x_range, y_range)
X0 = X0.flatten()
Y0 = Y0.flatten()
plt.scatter(X0, Y0, s=2, alpha = 0.5)

#Empty trajectory arrays
trajectories_x = np.empty((X0.shape[0],int(tau/dt_output)+1))
trajectories_y = np.empty((Y0.shape[0],int(tau/dt_output)+1))
    
#Integrate trajectories
for i in range(len(X0)):
    if i % 1000 == 0: print(str(i) + " / " + str(len(X0)))
    x0 = X0[i]
    y0 = Y0[i]
    sol = odeint(vel, [x0,y0], t)
    trajectories_x[i] = sol[:,0][::n_it]
    trajectories_y[i] = sol[:,1][::n_it]

np.savez("double_gyre_autonomous_trajectories", 
         drifter_longitudes = trajectories_x, drifter_latitudes = trajectories_y, drifter_time = [])