# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:29:46 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from create_network import trajectory_data
from datetime import datetime, timedelta


def plot_number_synchronous_trajectories():
    """
    Plot the number of synchronous drifter trajectories
    """
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID)
       
    drifter_data.start_end_times()
    start_times = drifter_data.start_times
    end_times = drifter_data.end_times
    
    t0_range = np.array([[datetime(y,m,1,0,0) for m in range(1,13)] for y in range(1989, 2020)]).flatten()
    days_range = np.arange(30,750,30)
    days_range = np.array([timedelta(days=int(d)) for d in days_range])
    counts = np.zeros((len(t0_range), len(days_range)))
    
    for k in range(len(time)):
        for i in range(len(t0_range)):    
            t0 = t0_range[i]
            for j in range(len(days_range)):
                d = days_range[j]
                if start_times[k]<=t0 and end_times[k] >=t0 + d:
                    counts[i,j]+=1
    
    x = range(len(t0_range))
    y = np.arange(30,750,30)
    x_ticks = [t0_range[i].strftime("%Y") for i in range(len(t0_range))]
    fig, ax = plt.subplots(figsize = (8,4))
    cmap = ax.pcolormesh(x, y, counts.transpose(), cmap = 'cividis')
    fig.colorbar(cmap)
    plt.xticks(x[::24], x_ticks[::24], rotation='vertical')
    plt.ylabel('Trajectory length (days)', size=11)
    plt.xlabel('Start time', size=11)
    plt.title('Number of synchronous drifter trajectories ', size=12)
    fig.tight_layout()
    plt.savefig('figures/number_of_synchronous_drifter_trajectories', dpi=300)