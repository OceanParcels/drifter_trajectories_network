"""
Script to process the 6 hourly drifter data:
    1. Limit drifters to those starting in North Atlantic
    2. Creates data set of similar trajectory lengths and uniform initial conditions
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
from create_network import trajectory_data
import random

def constrain_to_north_atlantic():
    """
    ---------------------------------------------------------------------
    Create file with only those drifters that start in the North Atlantic
    ---------------------------------------------------------------------
    """        

    """
    Load data
    """
    
    data_directory = '../drifter_data/drifters_6h_200318/'
    datafiles = [data_directory + 'buoydata_1_5000.dat',
                  data_directory + 'buoydata_5001_10000.dat',
                  data_directory + 'buoydata_10001_15000.dat',
                  data_directory + 'buoydata_15001_sep19.dat']
    
    d =  np.loadtxt(datafiles[0])
    ID = d[:,0]
    month = d[:,1]
    day = d[:,2]//1
    hour = (d[:,2]-d[:,2]//1)*24
    year = d[:,3]
    TIME = np.array([(datetime(int(year[i]), int(month[i]), int(day[i]), int(hour[i]))-datetime(1900, 1, 1, 0, 0)).total_seconds() for i in range(len(month))])
    LAT = d[:,4]
    LON = d[:,5]
    SST = d[:,6]
    V = d[:,9]
    print('First file loaded')
    
    for df in datafiles[1:]:
        print(df)
        d =  np.loadtxt(df)
        ID = np.append(ID, d[:,0])
        month = d[:,1]
        day = d[:,2]//1
        hour = (d[:,2]-d[:,2]//1)*24
        year = d[:,3]
        time = np.array([(datetime(int(year[i]), int(month[i]), int(day[i]), int(hour[i]))-datetime(1900, 1, 1, 0, 0)).total_seconds() for i in range(len(month))])
        TIME = np.append(TIME, time)
        LAT = np.append(LAT, d[:,4])
        LON = np.append(LON, d[:,5])
        SST = np.append(SST, d[:,6])
        V = np.append(V, d[:,9])
    
    LON[LON>180.]-=360.
    
    
    """
    Plot example trajectory
    """
    unique_ids = np.unique(ID)
    i = unique_ids[5000]
    lon = LON[ID == i]
    lat = LAT[ID == i]
    time = TIME[ID == i]
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawcoastlines()
    plt.title('An example drifter trajectory')
    xs, ys = m(lon, lat)
    m.scatter(xs, ys, s=2)
    
    
    """
    Get initial positions
    """
    lons_initial = np.array([LON[ID == i][0] for i in unique_ids])
    lats_initial = np.array([LAT[ID == i][0] for i in unique_ids])
     
    
    """
    Plot initial positions
    """
    plt.figure(figsize=(12,8))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawcoastlines()
    plt.title('Initial position whole data set')
    xs, ys = m(lons_initial, lats_initial)
    m.scatter(xs, ys, s=2)
    
    """
    Select drifter IDs corresponding to North Atlantic
    """
    selection0 = [i for i in range(len(lons_initial)) if (lats_initial[i]>0) and (lats_initial[i]<80) and (lons_initial[i]>-100) and (lons_initial[i]<20)]
    
    (lo0, la0) = (-5.49, 35.47)
    (lo1, la1) = (8., 58.9)
    a = (la1-la0)/(lo1-lo0)
    selection1 = [i for i in range(len(lons_initial)) if ((lats_initial[i] > la0 + a * (lons_initial[i]-lo0) and lons_initial[i]>lo0)  or (lons_initial[i]<=lo0) or (lats_initial[i]<la0-10) or (lats_initial[i]>65))]
        
    (lo0, la0) = (-103.5, 21.)
    (lo1, la1) = (-75., 4)
    a = (la1-la0)/(lo1-lo0)
    selection2 = [i for i in range(len(lons_initial)) if ((lats_initial[i] > la0 + a * (lons_initial[i]-lo0) and lons_initial[i]>lo0)) or lons_initial[i]>-60.]
        
    selection3 = [i for i in range(len(lons_initial)) if not ((lons_initial[i] <-84. and lats_initial[i] < 14.) or (lons_initial[i] <-90. and lats_initial[i] < 18.) or (lons_initial[i] <-76. and lats_initial[i] < 8.) or (lons_initial[i] <-78.2 and lats_initial[i] < 9.1 and lons_initial[i] > -80.6))]
    
    selection = list(set.intersection(set(selection0), set(selection1), set(selection2), set(selection3)))
    
    unique_ids_north_atlantic=unique_ids[selection]
    
    
    """
    Plot initial positions of North Atlantic drifters
    """
    lons_initial_north_atlantic = [np.array(lons_initial)[unique_ids == i] for i in unique_ids_north_atlantic]
    lats_initial_north_atlantic = [np.array(lats_initial)[unique_ids == i] for i in unique_ids_north_atlantic]
    
    plt.figure(figsize=(12,8))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawcoastlines()
    plt.title('Initial positions of drifters starting in North Atlantic')
    xs, ys = m(lons_initial_north_atlantic, lats_initial_north_atlantic)
    m.scatter(xs, ys, s=2)
    
    
    """
    Reduce data and save
    """
    lons_north_atlantic = [LON[ID == i] for i in unique_ids_north_atlantic]
    lats_north_atlantic = [LAT[ID == i] for i in unique_ids_north_atlantic]
    time_north_atlantic = [TIME[ID == i] for i in unique_ids_north_atlantic]
    SST_north_atlantic = [SST[ID == i] for i in unique_ids_north_atlantic]
    V_north_atlantic = [V[ID == i] for i in unique_ids_north_atlantic]
    
    assert(len(time_north_atlantic)==len(lons_north_atlantic))
    assert(len(lats_north_atlantic)==len(lons_north_atlantic))
    np.savez('drifter_data_north_atlantic/drifterdata_north_atlantic', lon = lons_north_atlantic, lat = lats_north_atlantic, 
              time = time_north_atlantic, ID = unique_ids_north_atlantic, SST = SST_north_atlantic,
              V = V_north_atlantic)
     

def check_data():
    """
    Check if saved data is correct
    """
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    
    t0 = [timedelta(seconds = time[0][i]) + datetime(1900, 1, 1, 0, 0) for i in range(len(time[0]))]
    print(t0)
    
    lon0 = [lon[i][0] for i in range(len(lon))]
    lat0 = [lat[i][0] for i in range(len(lon))]
    
    plt.figure(figsize=(12,8))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawcoastlines()
    plt.title('Initial positions of drifters starting in North Atlantic')
    xs, ys = m(lon0, lat0)
    m.scatter(xs, ys, s=2)


def create_coherent_monthly_drifter_data():
    """
    Create data set with drifters starting in a certain month and trajectories 
    of certain length. This only takes into account the month (not e.g. the first of the month).
    - It is assumed that the data is 6-hourly
    
    """
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    time_full = [np.array([timedelta(seconds = time[i][j]) + datetime(1900, 1, 1, 0, 0) 
                           for j in range(len(time[i]))]) for i in range(len(time))]
    
    for month_start in [1,4,7,10]:
        for trajectory_lenght_days in [60,120,180,360]:
            
            lons = []
            lats = []
            t = []
            trajectory_lenghts_n = trajectory_lenght_days * 4
            
            for i in range(len(lon)):
                if i%50==0: print(i)
                months = np.array([time_full[i][j].month for j in range(len(time_full[i]))])
                if np.any(months==month_start): 
                    i0 = np.argwhere(months==month_start)[0][0]
                    K = len(months[i0:])//trajectory_lenghts_n
                    for k in range(K):
                        lons += [lon[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
                        lats += [lat[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
                        t += [time_full[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
    
            np.savez('drifter_data_north_atlantic/time_coherent_drifters_month_start' + str(month_start) 
                     + 'length_days_' + str(trajectory_lenght_days), lon = lons, lat = lats, time=t)


def create_month_aggregated_coherent_drifter_data():
    
    """
    Same as create_coherent_monthly_drifter_data, but aggregated over all starting months
    """
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    time_full = [np.array([timedelta(seconds = time[i][j]) + datetime(1900, 1, 1, 0, 0) 
                           for j in range(len(time[i]))]) for i in range(len(time))]
    
    # for trajectory_lenght_days in [60,120,180,360]:    
    trajectory_lenght_days = 365
    lons = []
    lats = []
    t = []
    
    trajectory_lenghts_n = trajectory_lenght_days * 4
    
    for month_start in range(1,13):
        
        for i in range(len(lon)):
            if i%50==0: print(i)
            months = np.array([time_full[i][j].month for j in range(len(time_full[i]))])
            if np.any(months==month_start): 
                i0 = np.argwhere(months==month_start)[0][0]
                K = len(months[i0:])//trajectory_lenghts_n
                for k in range(K):
                    lons += [lon[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
                    lats += [lat[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
                    t += [time[i][i0 + trajectory_lenghts_n * k :i0+ + trajectory_lenghts_n * (k+1) ]]
    
    
    np.savez('drifter_data_north_atlantic/time_coherent_drifters_combined_'
                 + 'length_days_' + str(trajectory_lenght_days), drifter_longitudes = lons, drifter_latitudes = lats, 
                 drifter_time=t)


def create_uniformized_dataset(d_deg=1.):
    """
    Function to create a uniformized data set by selecting exactly one trajectory starting
    in each box of size d_deg
    """
    
    #Load data, but only daily positions, and plot
    drifter_data = trajectory_data.from_npz('drifter_data_north_atlantic/time_coherent_drifters_combined_length_days_365.npz',
                                            n_step=4)
    print('Size before: ', drifter_data.N)
    f, ax = plt.subplots(figsize=(10,10))
    drifter_data.compute_initial_symbols(d_deg = d_deg)
    drifter_data.plot_discretized_distribution(drifter_data.initial_distribution, ax,
                                               logarithmic = True)
    
    #Get initil symbols
    initial_symbols = drifter_data.initial_symbols
    unique_initial_symbols, initial_symbol_counts = np.unique(initial_symbols, return_counts=True)

    #Loop  over symbols and check where there is more than one initial conditions. 
    #In that case, randomly select one
    drifter_indices = []
    for i_s in unique_initial_symbols:
        drifter_indices_i_s = np.argwhere(initial_symbols==i_s)[:,0]
        drifter_index_select = random.choice(drifter_indices_i_s)
        drifter_indices += [drifter_index_select]
        
    drifter_data.restrict_to_subset(drifter_indices)    
    print('Size after: ', drifter_data.N)
    
    f, ax = plt.subplots(figsize=(10,10))
    
    drifter_data.compute_initial_symbols(d_deg = d_deg)
    drifter_data.plot_discretized_distribution(drifter_data.initial_distribution, ax,
                                               logarithmic = False)
    
    drifter_data.save_to_npz('drifter_data_north_atlantic/uniformized_dataset_365days_1deg_uniform')

