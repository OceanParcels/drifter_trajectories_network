# -*- coding: utf-8 -*-
"""
Plot drifter data set info
"""

import numpy as np
import matplotlib.pyplot as plt
from create_network import trajectory_data
from datetime import datetime, timedelta
import matplotlib.colors as colors


def plot_drifter_data():
    """
    Plot with total drifter measurement counts (map) and histogram
    """
    
    drifter_data = trajectory_data.from_npz('drifter_data_north_atlantic/drifterdata_north_atlantic.npz',
                                            set_nans=False)
    drifter_data.set_nans(N_nans = np.inf)
    
    #Total counts
    drifter_data.compute_symbolic_sequences(d_deg = 1.0, dt_days = 0.25)
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
    drifter_data.compute_initial_symbols(d_deg = 1.0)
    drifter_distribution0 = drifter_data.initial_distribution
    
    
    #Histogram of drifter lifetimes
    drifter_data.start_end_times()
    start_times = drifter_data.start_times
    end_times = drifter_data.end_times
    t_diff = np.array([(e-s).days for s,e in zip(start_times, end_times)])
    hist, x_hist = np.histogram(t_diff, bins=50, density=True)
    t_range = x_hist[0] + np.cumsum(np.diff(x_hist))
    
    #Length and possible start times color plot
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
    
    
    #With monthly start times
    c_month = np.zeros((12,24))
    for i in range(12):
        c_month[i] = np.sum(counts[i::12,:], axis=0)
        
        
    f = plt.figure(constrained_layout=True, figsize = (10,6))
    gs = f.add_gridspec(2, 3)
    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('a) drifter release', size=12)
    norm = colors.Normalize(0,10)
    drifter_data.plot_discretized_distribution(drifter_distribution0, ax1, land=True, cmap='cividis', 
                                               cbar_orientation ='vertical', logarithmic=False,
                                               norm=norm)
    
    ax2 = f.add_subplot(gs[0, 1])
    ax2.set_title('b) total counts', size=12)
    drifter_data.plot_discretized_distribution(drifter_distribution, ax2, land=True, cmap='cividis', 
                                               cbar_orientation ='vertical', logarithmic=False)
    
    ax3 = f.add_subplot(gs[0 ,2])
    ax3.bar(t_range[t_range<2000], hist[t_range<2000], width = np.diff(x_hist)[0], color='darkslategrey')
    ax3.set_title('c) drifter lifetimes (days)', size=12)
    ax3.grid(True)
    
    ax4 = f.add_subplot(gs[1, 0:2])
    cmap = ax4.pcolormesh(x, y, counts.transpose(), cmap = 'cividis')
    plt.colorbar(cmap, orientation = 'horizontal')
    plt.xticks(x[::24], x_ticks[::24], rotation='vertical')
    plt.ylabel('Trajectory length (days)', size=11)
    plt.xlabel('Start time', size=11)
    plt.title('d) number of synchronous drifter trajectories ', size=12)
    
    ax5 = f.add_subplot(gs[1, 2])
    m_range = [6,9,12]
    for m in m_range:
        ax5.plot(np.arange(1,13,1), c_month[:,m-1], 'o--', markersize=5, label=str(m))
    ax5.grid(True)
    ax5.set_xticks(np.arange(1,13,1))
    ax5.set_ylabel('# trajectories', size=11)
    ax5.set_xlabel('Start time (month)', size=11)
    ax5.set_title('e) starting month only', size=12)
    plt.legend(title = 'length (months)', loc='upper center', bbox_to_anchor=(0.5, -0.3),
              fancybox=True, shadow=True, ncol=5)
    
    plt.savefig('figures/drifter_dataset_info', dpi=300)

plot_drifter_data()