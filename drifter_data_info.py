# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:18:00 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from create_network import trajectory_data
from datetime import datetime, timedelta
import matplotlib.colors as colors

d_deg=[2.0]
domain_edges = (-100., 80, 0, 85) # full domain, to cover all points along any trajectory    
drifter_data = trajectory_data.from_npz('drifter_data_north_atlantic/drifterdata_north_atlantic.npz',
                                         time_interval_days=0.25, n_step = 1, domain_type = "north_atlantic_domain",
                                         domain_edges = domain_edges, set_nans=False)

drifter_data.set_nans(constrain_to_domain=True) #To enforce entire data on north atlantic
   
#Total counts
drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = 0.25)
sequence = drifter_data.symbolic_sequence
total_symbols = []
for s in sequence: total_symbols += list(s[:])
total_symbols = np.array(total_symbols).astype(int)
total_symbols = total_symbols[total_symbols>=0]
unique_symbols, counts = np.unique(total_symbols, return_counts=True)
drifter_distribution = np.zeros(drifter_data.n_horizontal)
drifter_distribution[unique_symbols] = counts
drifter_distribution = np.ma.masked_array(drifter_distribution, drifter_distribution==0)   

#Initial positions
drifter_data.compute_initial_symbols(d_deg = d_deg)
drifter_distribution0 = drifter_data.initial_distribution
    
#Length and possible start times color plot
drifter_data.start_end_times()
start_times = drifter_data.start_times
end_times = drifter_data.end_times
t0_range = np.array([[datetime(y,m,1,0,0) for m in range(1,13)] for y in range(1989, 2020)]).flatten()
days_range = np.arange(30,750,30)
days_range = np.array([timedelta(days=int(d)) for d in days_range])
counts = np.zeros((len(t0_range), len(days_range)))

for k in range(drifter_data.N):
    for i in range(len(t0_range)):    
        t0 = t0_range[i]
        for j in range(len(days_range)):
            d = days_range[j]
            if start_times[k]<=t0 and end_times[k] >=t0 + d:
                counts[i,j]+=1

x = range(len(t0_range))
y = np.arange(30,750,30)
x_ticks = [t0_range[i].strftime("%Y") for i in range(len(t0_range))]

f = plt.figure(constrained_layout=True, figsize = (6,5.5))
gs = f.add_gridspec(2, 2)

ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('a) total data', size=12)
norm = colors.Normalize(0,20000)
drifter_data.plot_discretized_distribution(drifter_distribution, ax1, land=True, cmap='plasma', 
                                           cbar_orientation ='vertical', logarithmic=False,
                                           norm = norm, extent='max')

norm = colors.Normalize(0,20)
ax2 = f.add_subplot(gs[0, 1])
ax2.set_title('b) drifter release', size=12)
drifter_data.plot_discretized_distribution(drifter_distribution0, ax2, land=True, cmap='plasma', 
                                           cbar_orientation ='vertical', logarithmic=False,
                                           norm=norm, extent = 'max')

ax4 = f.add_subplot(gs[1, 0:2])
cmap = ax4.pcolormesh(x, y, counts.transpose(), cmap = 'cividis')
plt.colorbar(cmap, orientation = 'horizontal')
plt.xticks(x[::24], x_ticks[::24], rotation='vertical')
plt.ylabel('Trajectory length (days)', size=10)
plt.xlabel('Start time', size=11)
plt.title('c) number of synchronous drifter trajectories ', size=12)

plt.savefig('figures/north_atlantic/drifter_dataset_info', dpi=300)