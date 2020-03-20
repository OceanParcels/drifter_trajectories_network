# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:19:29 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.basemap import Basemap

class domain(object):    
    """
    Class to contain the domain and grids for plotting
    """
    
    def __init__(self, minlon, maxlon, minlat, maxlat, domaintype):
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat
        self.domaintype = domaintype
       
    def setup_grid(self, d_deg):
        self.d_deg = d_deg
        self.Lons_edges = np.linspace(self.minlon,self.maxlon,int((self.maxlon-self.minlon)/d_deg)+1) 
        self.Lats_edges = np.linspace(self.minlat,self.maxlat,int((self.maxlat-self.minlat)/d_deg)+1) 
        self.Lons_centered=np.diff(self.Lons_edges)/2+self.Lons_edges[:-1]
        self.Lats_centered=np.diff(self.Lats_edges)/2+self.Lats_edges[:-1]
        self.lon_bins_2d, self.lat_bins_2d = np.meshgrid(self.Lons_edges, self.Lats_edges)   
    
    @classmethod
    def north_atlantic_domain(cls):
        (minlon, maxlon, minlat, maxlat) = (-120., 80., 0., 90.)
        return cls(minlon, maxlon, minlat, maxlat, domaintype='north_atlantic')


class trajectory_data(object):
    
    def __init__(self, drifter_longitudes, drifter_latitudes, 
                 drifter_time, drifter_id, data_description = "no description",
                 set_nans = True):
        self.drifter_longitudes = drifter_longitudes
        self.drifter_latitudes = drifter_latitudes
        self.drifter_time = drifter_time
        self.drifter_id = drifter_id
        self.N = len(drifter_id)
        self.data_description = data_description
        if set_nans:
            self.set_nans()
            self.nans_set = True
        else:
            self.nans_set = False
    
        self.discretized = False
    
    def set_nans(self):
        print('Setting wrong data points to nan')
        
        n_data_errors = 0
        
        for i in range(self.N):
            n_data_errors += len(self.drifter_longitudes[i][self.drifter_longitudes[i]>180])
            self.drifter_latitudes[i][self.drifter_longitudes[i]>180]=np.nan
            self.drifter_longitudes[i][self.drifter_longitudes[i]>180]=np.nan
            
        print("found " + str(n_data_errors) + " nans")            
        # self.drifter_longitudes[self.drifter_longitudes>180]=np.nan
        # self.drifter_latitudes[self.drifter_longitudes>180]=np.nan
        self.nans_set = True
    
    def distance_on_sphere(self, i1, i2, trajectory_type):
        
        if trajectory_type == 'time_coherent':
            lons1, lats1 = self.drifter_longitudes[i1], self.drifter_latitudes[i1]
            lons2, lats2 = self.drifter_longitudes[i2], self.drifter_latitudes[i2]
        elif trajectory_type == 'time_incoherent':
            time1, time2 = self.drifter_time[i1], self.drifter_time[i2]
            
            min_time = np.max([np.min(time1), np.min(time2)])
            max_time = np.min([np.max(time1), np.max(time2)])
        
            interval_1 = np.argwhere((time1 >= min_time) & (time1 < max_time))[:,0]
            interval_2 = np.argwhere((time2 >= min_time) & (time2 < max_time))[:,0]
            assert(len(interval_1) == len(interval_2))
            
            lons1, lats1 = self.drifter_longitudes[i1][interval_1], self.drifter_latitudes[i1][interval_1]
            lons2, lats2 = self.drifter_longitudes[i2][interval_2], self.drifter_latitudes[i2][interval_2]
            time1, time2 = self.drifter_time[i1][interval_1], self.drifter_time[i2][interval_2]
            assert(np.all(time1 - time2 ==0.)) #See if this is too strict

            # if not np.all(time1 - time2 ==0.):
            #     print("Time vectors different. Returning nan.")
            #     return np.nan
        
        assert(len(lons1) == len(lons2))
        
        if len(lons1)==0:
            return np.nan
        else:
            r = 6371.
            a = np.pi/180.
            arg = np.sin(a * lats1) * np.sin(a * lats2) + np.cos(a * lats1) * np.cos(a * lats2) * np.cos(a * (lons1-lons2))
            
            for i in range(len(arg)):
                if (np.abs(arg[i]) - 1.)>1e-5: arg[i]=1
            
            d = r * np.arccos(arg)
            return d
        
    
    def construct_mindist_network(self, trajectory_type = 'time_incoherent', 
                                  output_name='noname_count_matrix'):
        """
        data_type options 'time_coherent' and 'time_incoherent':
            - time_coherent assumes that the time of sampling is the 
            same for all drifters, i.e. it basically disregards the time info
            - time_incoherent takes into account that different trajectories have
            different time data points
        """
        
        if not self.nans_set:
            self.set_nans()
        
        A = np.empty((self.N, self.N))
        
        for i in range(self.N):
            if i%50 ==0: print(str(i) + " / " + str(self.N))
            for j in range(i+1, self.N):
                A[i,j] = np.nanmin(self.distance_on_sphere(i,j, trajectory_type))
    
        self.A = A + A.transpose()
        np.savez(output_name, count_matrix = A, description = self.data_description)
        
    
    def set_discretizing_values(self, d_deg=0.5):
        """
        Set cell size for matrix computation
        """
        self.d_deg = d_deg
        self.domain = domain.north_atlantic_domain()
        self.n_lons  = int((self.domain.maxlon - self.domain.minlon)/d_deg)
        self.n_lats  = int((self.domain.maxlat - self.domain.minlat)/d_deg)
        self.n_horizontal   = self.n_lons * self.n_lats        
        self.domain.setup_grid(d_deg)
        self.discretized = True 
    
    
    def coords_to_matrixindex2D(self, coord_2D):
        (lon, lat) = coord_2D
        if np.isnan(lon) or np.isnan(lat):
            return np.nan
        else:
            index_2D    = int((lat - self.domain.minlat)//self.d_deg * self.n_lons + (lon - self.domain.minlon)//self.d_deg)
            return index_2D


    def compute_symbolic_sequences(self, d_deg=0.5, dt_days=5):
        
        self.set_discretizing_values(d_deg)
        dt_indices = dt_days * 4 #as data is every 6 hours
        
        self.t_diffs = np.array([np.mean(np.diff(self.drifter_time[i][::dt_indices]))/(86400) for i in range(self.N)])
        if np.any(np.abs(self.t_diffs-dt_days)>0.01):
            print("time differences are not consistent! Consider \
                  discarding these drifters!")
            print(self.t_diffs[np.abs(self.t_diffs-dt_days)>0.01])
        
        self.symbolic_sequence = []
        
        for k in range(dt_indices):
            self.symbolic_sequence += [np.array([self.coords_to_matrixindex2D((lo,la)) for lo, la in 
                                      zip(self.drifter_longitudes[i][k::dt_indices],
                                          self.drifter_latitudes[i][k::dt_indices])]) for i in range(self.N)]
        
    def compute_transition_matrix(self):
        
        initial_indices = []
        final_indices = []
        
        symbolic_sequence = self.symbolic_sequence
        
        for i in range(len(symbolic_sequence)):
            initial_indices += list(symbolic_sequence[i][:-1])
            final_indices += list(symbolic_sequence[i][1:])

        #constrain to domain. This also removes nans   
        initial_indices = np.array(initial_indices)
        final_indices = np.array(final_indices)
        indices = (initial_indices>0) & (final_indices>0)
        initial_indices = initial_indices[indices]
        final_indices = final_indices[indices]
        
        np.save('initial_indices', initial_indices)
        np.save('final_indices', final_indices)
        
        print('Computing transition and count matrices')
        transitions = np.ones(len(initial_indices))
        from scipy.sparse import coo_matrix
        W = coo_matrix((transitions, (initial_indices, final_indices)), shape=(self.n_horizontal, self.n_horizontal)).tocsr()
        sparse.save_npz('count_matrix', W)
        d = np.array(sparse.csr_matrix.sum(W, axis=1))[:,0]
        np.save('initial_distribution', d)
        from sklearn.preprocessing import normalize
        P = normalize(W, norm='l1', axis=1)
        sparse.save_npz('transition_matrix', P)
        
        
    def volume_field(self):
        a = 6371000.
        diff_sin_ = np.diff(np.sin(self.domain.Lats_edges * np.pi/180.))
        A_lat = a**2 * diff_sin_ * (self.d_deg * np.pi / 180.)
        area = np.array([list(A_lat)])
        A_horizontal = np.array([list(np.repeat(area, self.n_lons, axis=0).transpose())])        
        return A_horizontal[0]  

    def plot_discretized_distribution(self, v, cmap = None, land = False, 
                                      title = 'notitle', colbar=True):
        """
        Plot 3D field section on horizontal
        - v: field to plot
        """

        #convert to 3-dimensional vector if needed
        if v.ndim == 1: d2d = v.reshape(self.n_lats, self.n_lons)
        else: d2d = v
        
        #plotting
        if cmap == None: cmap = 'plasma'
      
        fig = plt.figure(figsize = (8,10))

        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat,urcrnrlat=self.domain.maxlat, llcrnrlon=self.domain.minlon,urcrnrlon=self.domain.maxlon,resolution='c')
        m.drawparallels([15,30,45,60,75], labels=[True, False, False, True], linewidth=1.8, size=15)
        m.drawmeridians([-70,-40,-10,20,50], labels=[False, False, False, True], linewidth=1.8, size=15)
        m.drawcoastlines()
         
        if land: m.fillcontinents(color='dimgray')
        xs, ys = m(self.domain.lon_bins_2d, self.domain.lat_bins_2d)
        p=plt.pcolormesh(xs, ys, d2d, rasterized=True, cmap = cmap) #, vmin=-40, vmax=1)
        plt.title(title, size=20, y=1.01)

        if colbar: 
            cbar = fig.colorbar(p, shrink=.2)
            cbar.ax.tick_params(labelsize=14) 

        plt.show()    
        
    
def construct_count_matrix_drifters():
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    description = "first test"
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                   drifter_time = time, drifter_id = ID, 
                                   data_description = description)
    
    drifter_data.construct_mindist_network(trajectory_type = 'time_incoherent', 
                                           output_name = 'first_try')


def compute_drifter_transitionmatrix():
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    description = "first test"
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID, 
                                    data_description = description)
    drifter_data.compute_symbolic_sequences()
    drifter_data.symbolic_sequence
    
    print("computing transition matrix")
    drifter_data.compute_transition_matrix()


