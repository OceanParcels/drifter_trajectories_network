"""
Script to process the 6h drifter data
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap


def constrain_to_north_atlantic():
        
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
    Plot iniial positions
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
     

"""
Check if saved data is correct
"""
def check_data():
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
    