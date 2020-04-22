"""
Script to produce trajectories for the double gyre flow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

A = 0.25
delta = 0.25
omega = 2 * np.pi

def vel(y,t):
    x1 = y[0]
    x2 = y[1]
    f = delta * np.sin(omega * t) * x1**2 + (1-2 * delta * np.sin(omega * t)) * x1
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * x2)
    v = np.pi * A *np.cos(np.pi * f) * np.sin(np.pi * x2) * (2 * delta * np.sin(omega * t ) * x2 +  1-2*delta *np.sin(omega * t))    
    return [u,v]

tau = 20
dt_output = 0.1
dt_int = 0.001
n_it = int(dt_output / dt_int)

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

np.savez("double_gyre_trajectories_np_" + str(len(x_range)*len(y_range)) + '_tau_' + str(tau) + '_dt_' + str(dt_output), 
         drifter_longitudes = trajectories_x, drifter_latitudes = trajectories_y, drifter_time = [])