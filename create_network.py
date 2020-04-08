# -*- coding: utf-8 -*-
"""
Classes to process drifter data:
    - Class domain: specify the region of interest (here North Atlantic).
    - Class trajectory data: handle lon/lat/time drifter data and construct networks from them
    - Class undirected_network: network analysis, mostly spectral clustering, of an undirected network

Notes:
    - If applied to other regions than north atlantic, adjust the data set and the domain
"""

import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from scipy.sparse import coo_matrix
from datetime import datetime, timedelta
import scipy.sparse
import networkx as nx
from sklearn.cluster import KMeans


class domain(object):
    """
    Class containing the domain and grids for plotting.
    """
    
    def __init__(self, minlon, maxlon, minlat, maxlat,
                 minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                 parallels, meridians):
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat
        
        #For plotting
        self.minlon_plot = minlon_plot
        self.maxlon_plot = maxlon_plot
        self.minlat_plot = minlat_plot
        self.maxlat_plot = maxlat_plot
        self.parallels = parallels
        self.meridians = meridians
        
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
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-100., 40., 0., 85.)
        (parallels, meridians) = ([20,40,60,80],[-70,-40,-10,20,50])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)


class trajectory_data(object):
    """
    Class to
    - handle drifter data
    - compute the networks used for clustering
    - plot data, e.g. a field over North Atlantic Ocean
    """
    
    def __init__(self, drifter_longitudes=[], drifter_latitudes=[], 
                 drifter_time=[], drifter_id=[], time_0 =  datetime(1900, 1, 1, 0, 0), 
                 time_interval_days=0.25, set_nans = True):
        """
        Parameters:
        - drifter_longitudes: format is len-N list of numpy arrays, each array containing the data of a trajectory.
        Similar for drifter_latitudes, drifter_time (seconds since time_0).
        - drifter_id: len-N list of drifter ID's. ID is not used at any point in the analysis.
        - time_0: reference time. time unit is in seconds since then.
        - time_interval_days: this is the time step between two measurements. Default is 6 hours
        - set_nans: Some drifters have erroneous data. If True, this data is set to nan.
        """

        self.drifter_longitudes = drifter_longitudes
        self.drifter_latitudes = drifter_latitudes
        self.drifter_time = drifter_time
        self.drifter_id = drifter_id
        self.N = len(drifter_longitudes)
        self.time_interval_days = time_interval_days
        self.time_0 = time_0
        self.domain = domain.north_atlantic_domain()
        
        if set_nans:
            self.set_nans()
            self.nans_set = True
        else:
            self.nans_set = False
    
        self.discretized = False #True of partition into bins is defined
    
    
    @classmethod
    def from_npz(cls, filename, set_nans = True, time_interval_days=None, n_step=1):
        """
        Load data from .npz file:
            - filename has to contain: drifter_longitudes, drifter_latitudes, drifter_time
            - time_interval_days: the days between different data points of the
            - n_step: can restrict to every n_step'th data point. E.g. if data is 
            6-hourly (time_interval_days=0.25) and n_step=4, then trajectory_data object will be daily
        """
        
        if time_interval_days==None: 
            time_interval_days = 0.25
        
        data = np.load(filename, allow_pickle = True)
        lon = data['drifter_longitudes']
        lat = data['drifter_latitudes']
        time = data['drifter_time']
        
        lon = [lo[::n_step] for lo in lon]
        lat = [la[::n_step] for la in lat]
        time = [t[::n_step] for t in time]
        
        return cls(drifter_longitudes = lon, drifter_latitudes=lat, drifter_time = time,
                   time_interval_days=time_interval_days*n_step, set_nans = set_nans)
        
    
    def set_nans(self, N_nans=0):
        """
        The drifter data contains some empty data (where lon / lat are >600).
        Parameters:
            - N_nans: number of allowed nans per trajectory. Default is no nan.
        """
        
        print('Setting erroneous lon/lat  data points to nan')
        n_data_errors = 0
        remove_indices = []
        for i in range(self.N):
            n_data_errors += len(self.drifter_longitudes[i][self.drifter_longitudes[i]>180])
            self.drifter_latitudes[i][self.drifter_longitudes[i]>180]=np.nan
            self.drifter_longitudes[i][self.drifter_longitudes[i]>180]=np.nan
            if len(np.argwhere(np.isnan(self.drifter_longitudes[i])))>N_nans:
                remove_indices += [i]
        
        print('Number of trajectories containing nans:', len(remove_indices))
        self.drifter_latitudes = np.delete(self.drifter_latitudes, remove_indices, axis=0)
        self.drifter_longitudes = np.delete(self.drifter_longitudes, remove_indices, axis=0)
        self.drifter_time = np.delete(self.drifter_time, remove_indices, axis=0)
        self.N = len(self.drifter_longitudes)        
        self.nans_set = True
    
    
    def start_end_times(self):
        """
        Compute start and end times of drifters in datetime format
        """
        self.start_times = [timedelta(seconds = self.drifter_time[i][0]) + self.time_0 for i in range(len(self.drifter_time))]
        self.end_times = [timedelta(seconds = self.drifter_time[i][-1]) + self.time_0 for i in range(len(self.drifter_time))]
        self.trajectory_lengths = np.array([(self.end_times[i] - self.start_times[i]).days for i in range(len(self.end_times))])


    def distance_on_sphere(self, i1, i2):
        """
        Function to return a vector of distances (km) for two drifters with indices i1 and i2.
        Assumed is that the vectors are defined for the same times.
        """
        lats1 = self.drifter_latitudes[i1]
        lons1 = self.drifter_longitudes[i1]
        lats2 = self.drifter_latitudes[i2]
        lons2 = self.drifter_longitudes[i2]
    
        r = 6371.
        a = np.pi/180.
        arg = np.sin(a * lats1) * np.sin(a * lats2) + np.cos(a * lats1) * np.cos(a * lats2) * np.cos(a * (lons1-lons2))
        
        for i in range(len(arg)):
            if (np.abs(arg[i]) - 1.)>1e-5: arg[i]=1
        
        d = r * np.arccos(arg)
            
        return d
        

    def distance_on_sphere_time_incoherent(self, i1, i2):
        """
        Function to return a vector of distances (km) for two drifters with indices i1 and i2.
        The distances are only computed for the temporal overlap of the drifter data. If they
        do not overlap, nan is returned.
        
        """
        time1, time2 = self.drifter_time[i1], self.drifter_time[i2]
        
        min_time = np.max([np.min(time1), np.min(time2)])
        max_time = np.min([np.max(time1), np.max(time2)])
    
        interval_1 = np.argwhere((time1 >= min_time) & (time1 < max_time))[:,0]
        interval_2 = np.argwhere((time2 >= min_time) & (time2 < max_time))[:,0]
        assert(len(interval_1) == len(interval_2))
        
        lons1, lats1 = self.drifter_longitudes[i1][interval_1], self.drifter_latitudes[i1][interval_1]
        lons2, lats2 = self.drifter_longitudes[i2][interval_2], self.drifter_latitudes[i2][interval_2]
        time1, time2 = self.drifter_time[i1][interval_1], self.drifter_time[i2][interval_2]
        assert(np.all(time1 - time2 ==0.)) #Check that the time intervals are the same
        assert(len(lons1) == len(lons2))
        
        if len(lons1)==0: #no overlap
            return np.nan
        else:
            r = 6371.
            a = np.pi/180.
            arg = np.sin(a * lats1) * np.sin(a * lats2) + np.cos(a * lats1) * np.cos(a * lats2) * np.cos(a * (lons1-lons2))
            
            for i in range(len(arg)):
                if (np.abs(arg[i]) - 1.)>1e-5: arg[i]=1
            
            d = r * np.arccos(arg)
            return d
        
    
    def compute_M(self, distance_function='distance_on_sphere'):
        """
        Function to compute NxN network containing the minimum distance of two drifters.
        - distance_function: 'distance_on_sphere_time_incoherent', 'distance_on_sphere'
        """
        
        distance_function = getattr(self, distance_function)
        
        if not self.nans_set:
            self.set_nans()
        
        A = np.empty((self.N, self.N))
        print(A.shape)
        for i in range(self.N):
            if i%50 ==0: print(str(i) + " / " + str(self.N))
            for j in range(i, self.N):
                A[i,j] = np.nanmin(distance_function(i,j))
                
        self.M_mindist = A


    def compute_T(self, distance_function='distance_on_sphere'):
        """
        Function to compute NxN network containing the minimum distance of two drifters.
        - distance_function: 'distance_on_sphere_time_incoherent', 'distance_on_sphere'
        """
        
        distance_function = getattr(self, distance_function)
        
        if not self.nans_set:
            self.set_nans()
        
        A = np.empty((self.N, self.N))
        print(A.shape)
        for i in range(self.N):
            if i%50 ==0: print(str(i) + " / " + str(self.N))
            for j in range(i, self.N):
                A[i,j] = np.nanmean(distance_function(i,j))
                
        self.T_mean_distances = A



    def set_discretizing_values(self, d_deg=0.5):
        """
        Set cell size for matrix computation
        """
        
        self.d_deg = d_deg
        self.n_lons  = int((self.domain.maxlon - self.domain.minlon)/d_deg)
        self.n_lats  = int((self.domain.maxlat - self.domain.minlat)/d_deg)
        self.n_horizontal   = self.n_lons * self.n_lats        
        self.domain.setup_grid(d_deg)
        self.discretized = True 
    
    
    def restrict_to_subset(self, indices):
        """
        Limit drifter data to a subset, labelled by indices
        """
        
        self.drifter_longitudes = self.drifter_longitudes[indices]
        self.drifter_latitudes = self.drifter_latitudes[indices]
        self.drifter_time = self.drifter_time[indices]
        self.N = len(indices)
        
        
    def save_to_npz(self, filename):
        np.savez(filename, drifter_longitudes = self.drifter_longitudes,
                 drifter_latitudes = self.drifter_latitudes, drifter_time = self.drifter_time)
    
    
    def coords_to_matrixindex2D(self, coord_2D):
        """
        Function to compute the symbol for a certain binning, from a given (lon, lat) coordinate
        """
        
        (lon, lat) = coord_2D
        if np.isnan(lon) or np.isnan(lat):
            return np.nan
        else:
            index_2D    = int((lat - self.domain.minlat)//self.d_deg * self.n_lons + (lon - self.domain.minlon)//self.d_deg)
            return index_2D
    

    def compute_symbolic_sequences(self, d_deg=1.0, dt_days=60):
        """
        For transition matrix: Function to compute, for each drifter, the symbolic sequence given the 
        cell size of d_deg and time interval of dt_days. See respective section
        in the paper for more detailed explanation.
        """
        
        self.set_discretizing_values(d_deg)
        self.dt_days = dt_days
        dt_indices = int(dt_days / self.time_interval_days)
        
        self.symbolic_sequence = []        
        for k in range(dt_indices):
            self.symbolic_sequence += [np.array([self.coords_to_matrixindex2D((lo,la)) for lo, la in 
                                      zip(self.drifter_longitudes[i][k::dt_indices],
                                          self.drifter_latitudes[i][k::dt_indices])]) for i in range(self.N)]
    
    
    def compute_initial_symbols(self, d_deg=1.0):
        """
        Only compute initial symbols and distribution for plotting
        """
        self.set_discretizing_values(d_deg)
        self.initial_symbols = np.array([self.coords_to_matrixindex2D((lo[0],la[0])) for lo, la in zip(self.drifter_longitudes,
                                                                                              self.drifter_latitudes)])
        unique_symbols, counts = np.unique(self.initial_symbols, return_counts=True)
        
        d = np.zeros(self.n_horizontal)
        d[unique_symbols] = counts
        self.initial_distribution = d
    

    def compute_P(self, matrix_type='killed_Markov'):
        """
        Function to compute the transition matrix P, and the cell connection matrix, C.
        Note that the domain is chosen such that drifters that leave it have negative bin index,
        which we use to define it as Markovian drifter or not.
        """
        
        print('Computing transition matrix for type: ', matrix_type)
        
        initial_indices = []
        final_indices = []
        symbolic_sequence = self.symbolic_sequence
        
        for i in range(len(symbolic_sequence)):
            initial_indices += list(symbolic_sequence[i][:-1])
            final_indices += list(symbolic_sequence[i][1:])
        
        initial_indices = np.array(initial_indices)
        final_indices = np.array(final_indices)
            
        if matrix_type == 'Markov': #constrain to domain. This also removes nans   
            indices = (initial_indices>0) & (final_indices>0)
            initial_indices = initial_indices[indices]
            final_indices = final_indices[indices]
        elif matrix_type == 'killed_Markov':
            indices = (initial_indices>0) & (~np.isnan(final_indices))
            initial_indices = initial_indices[indices]
            final_indices = final_indices[indices]
        else:
            print('No known type specified')
        
        print('Computing transition and count matrices')
        transitions = np.ones(len(initial_indices))
        
        final_indices[final_indices<0] = self.n_horizontal
        W = coo_matrix((transitions, (initial_indices, final_indices)), shape=(self.n_horizontal+1, self.n_horizontal+1)).tocsr()
        d = np.array(sparse.csr_matrix.sum(W, axis=1))[0:self.n_horizontal,0]
        W = W[0:self.n_horizontal,:][:,0:self.n_horizontal]
        
        D_inv = sparse.diags([1./di if di>0 else 1. for di in d])
        P =  (W.transpose().dot(D_inv)).transpose()
        
        if matrix_type == 'Markov':
            s = np.array(sparse.csr_matrix.sum(P, axis=1))[:,0]
            assert(np.all(s[s>0]>1.-1e-9))
                    
        self.initial_indices = initial_indices
        self.final_indices = final_indices
        self.W = W
        self.d = d
        self.P = P
            
    
    def surface_area_field(self):
        """
        Function to return the surface area (km2) of the chosen cells as a vector
        """
        a = 6371.
        diff_sin_ = np.diff(np.sin(self.domain.Lats_edges * np.pi/180.))
        A_lat = a**2 * diff_sin_ * (self.d_deg * np.pi / 180.)
        area = np.array([list(A_lat)])
        A_horizontal = np.array([list(np.repeat(area, self.n_lons, axis=0).transpose())])        
        return A_horizontal[0]  


    def scatter_initial_position(self, ax, indices=None):
        """
        Scatter initial positions (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        lon_reduced = self.drifter_longitudes[indices]
        lat_reduced = self.drifter_latitudes[indices]
        lonplot = [lon_reduced[i][0] for i in range(len(lon_reduced))]
        latplot = [lat_reduced[i][0] for i in range(len(lat_reduced))]        
        xs, ys = m(lonplot, latplot)
        ax.scatter(xs, ys, s=2, alpha = 0.5)


    def scatter_trajectories(self, ax, indices):
        """
        Scatter whole trajectories (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        lon_reduced = self.drifter_longitudes[indices]
        lat_reduced = self.drifter_latitudes[indices]
        
        lonplot = lon_reduced[0]
        latplot = lat_reduced[0]
        
        for i in range(1, len(lon_reduced)):
            lonplot = np.append(lonplot, lon_reduced[i])
            latplot = np.append(latplot, lat_reduced[i])
        
        xs, ys = m(lonplot, latplot)
        ax.scatter(xs, ys, s=2, alpha = 0.5)
        

    def trajectories_density(self, indices, nmax=None):
        """
        Function to return a density field defined by all points on the trajectory
        """
        seq = self.symbolic_sequence
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        s = np.array(seq[indices[0]][:nmax]).astype(int)
        
        for i in indices[1:]:
            s = np.append(s, seq[i][:nmax].astype(int))
        
        s = s[s>=0]
        s_unique, s_counts = np.unique(s[~np.isnan(s)], return_counts=True)
        
        d = np.zeros(self.n_horizontal)
        d[s_unique] = s_counts
        
        self.d_cluster = d
        

    def scatter_initial_position_with_labels(self, ax, labels):
        """
        Scatter initial positions with color map given by labels
        """
        lon_plot = np.array([lo[0] for lo in self.drifter_longitudes])
        lat_plot = np.array([lo[0] for lo in self.drifter_latitudes])
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        xs, ys = m(lon_plot, lat_plot)
        p = ax.scatter(xs, ys, s=4, c=labels)
        plt.colorbar(p, shrink=.6, aspect=25)
        

    def scatter_trajectories_with_labels(self, ax, labels, random_sample=100, seed=None):
        """
        Scatter entire trajectories with color map given by labels.
        - random_sample: size of a random subsample to accelerate plotting
        """
        np.random.seed(seed)
        indices = np.random.randint(0, len(self.drifter_longitudes), size=random_sample)
        labels = labels[indices]
        lon_reduced = self.drifter_longitudes[indices]
        lat_reduced = self.drifter_latitudes[indices]
        
        lonplot = lon_reduced[0]
        latplot = lat_reduced[0]
        c = np.array([labels[0]] * len(lon_reduced[0]))
        
        for i in range(1, len(lon_reduced)):
            lonplot = np.append(lonplot, lon_reduced[i])
            latplot = np.append(latplot, lat_reduced[i])
            c = np.append(c, np.array([labels[i]] * len(lon_reduced[i])))
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        indices_plot = np.random.shuffle(np.array(range(len(lonplot))))
        xs, ys = m(lonplot, latplot)
        p = ax.scatter(xs[indices_plot], ys[indices_plot], s=2, c=c[indices_plot],  alpha = 0.1)
        plt.colorbar(p, shrink=.6, aspect=25)
        

    def plot_discretized_distribution(self, v, ax, cmap = None, norm=None, land = False, 
                                      title = 'notitle', colbar=True, 
                                      cbartitle = None, logarithmic = False,
                                      cbar_orientation = 'horizontal'):
        """
        Plot 2D field
        - v: field to plot
        - ax: axis object to plot in
        """

        #convert to 3-dimensional vector if needed
        if v.ndim == 1: d2d = v.reshape(self.n_lats, self.n_lons)
        else: d2d = v
        
        #plotting
        if cmap == None: cmap = 'plasma'
        if norm == None: norm= colors.Normalize(vmin=np.ma.min(d2d), vmax=np.ma.max(d2d))
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()

        if land: m.fillcontinents(color='dimgray')
        xs, ys = m(self.domain.lon_bins_2d, self.domain.lat_bins_2d)
        if logarithmic:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap,  norm=colors.LogNorm())
        else:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap, norm=norm)# , vmin=-20, vmax=1)

        if colbar: 
            if cbar_orientation == 'horizontal':
                cbar = plt.colorbar(p, shrink=.6, aspect=10, orientation="horizontal")
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=40)
            cbar.ax.tick_params(labelsize=9) 
            if cbartitle is not None:
                cbar.set_label(cbartitle, size=9)  
        
        


class undirected_network(object):
    
    def __init__(self, adjacency_matrix, cluster_indices=np.array([]), cluster_volume=np.array([]), cluster_label = 0):
        
        """
        adjacency_matrix: format sparse.csr_matrix. If it is not symmetric it is symmetrized.
        region_indices: indices corresponding to network domain.
        cluster_volume: vector of volume of the nodes inside the cluster.
        """
        if len(cluster_indices)==0:
            cluster_indices = np.array(range(adjacency_matrix.shape[0]))
        if len(cluster_volume)==0:
            cluster_volume = np.array(sparse.csr_matrix.sum(adjacency_matrix, axis=1))[:,0]
        
        self.adjacency_matrix = adjacency_matrix
        self.cluster_indices = cluster_indices
        self.cluster_volume = cluster_volume
        self.N = adjacency_matrix.shape[0]
        self.cluster_label = cluster_label
        self.rho = np.sum(self.adjacency_matrix)/np.sum(self.cluster_volume)
        assert(len(cluster_indices) == self.adjacency_matrix.shape[0])
        assert(len(cluster_volume) == len(cluster_indices))
        print('Construct undirected network.')
    
    
    def __del__(self):
        print('Adjacency matrix object deleted')

      
    def connected_components(self):

        """
        Determine connected components
        """
        
        G = nx.from_scipy_sparse_matrix(self.adjacency_matrix, create_using = nx.Graph())
        components = np.array(list(nx.connected_components(G)))
        component_lengths = np.array([len(s) for s in components])
        component_inds = np.argsort(component_lengths)[::-1]
        components_sorted = components[component_inds]
        component_lengths = np.array([len(c) for c in components_sorted])
        
        print('Component lengths (>1):')
        print(component_lengths[component_lengths>1])
        
        n = int(input('Please specify how many components you want to keep: '))
        components_sorted = components_sorted[:n]
        
        
        sub_networks = []
        
        for i in range(len(components_sorted)):
            inds = list(components_sorted[i])
            sub_adjacency_matrix = self.adjacency_matrix[inds, :][:, inds]
            sub_cluster_indices = self.cluster_indices[inds]
            sub_cluster_volume = self.cluster_volume[inds]
            sub_cluster_label = self.cluster_label
            sub_networks.append(undirected_network(sub_adjacency_matrix, sub_cluster_indices, 
                                                   sub_cluster_volume, sub_cluster_label))
        
        self.sub_networks = sub_networks
        

    def compute_laplacian_spectrum(self, K=20):
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags(1./np.sqrt(d))
        L = sparse.identity(self.N) - (D_sqrt_inv.dot(self.adjacency_matrix)).dot(D_sqrt_inv)
        print('Computing spectrum of symmetric normalized Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        plt.plot(w, 'o')
        plt.title('Eigenvalues of symmetric normalized Laplacian')
        plt.grid(True)
        plt.show()
        
        self.Lsym_eigenvalues = w
        self.Lsym_eigenvectors = D_sqrt_inv.dot(v)


    def compute_unnormalized_laplacian_spectrum(self, K=20):
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D = scipy.sparse.diags(d)
        L = D - self.adjacency_matrix
        print('Computing spectrum of symmetric normalized Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        plt.plot(w, 'o')
        plt.title('Eigenvalues of symmetric normalized Laplacian')
        plt.grid(True)
        plt.show()
        
        self.L_eigenvalues = w
        self.L_eigenvectors = v

    
    def cluster_kmeans(self, K, rs=0, method = 'ncut'):
        if method == 'ncut':
            X = self.Lsym_eigenvectors[:,:K]        
        elif method == 'cut':
            X = self.L_eigenvectors[:,:K]            
        kmeans = KMeans(n_clusters=K, random_state=rs).fit(X)        
        self.kmeans_labels = kmeans.labels_


    # def cluster_directional_cosine(self, K, rs=0): #maybe remove
    #     X = np.array(self.Lsym_eigenvectors[:,:K])
    #     X_normalized = np.array([X[i] / np.sqrt(X[i].dot(X[i])) for i in range(len(X))])
        
    #     Z = X_normalized.dot(X_normalized.transpose())
    #     kmeans = KMeans(n_clusters=K, random_state=rs).fit(Z)        
    #     self.directional_cosine_labels = kmeans.labels_
        

    def drho_split(self, indices_1, indices_2):
        """
        If we propose to split a cluster, this returns the changes in the coherence ratio for a split into
        indices_1 and indices_2
        """
        cluster_volume_1 = np.sum(self.cluster_volume[indices_1])
        cluster_volume_2 = np.sum(self.cluster_volume[indices_2])
        stays_in_1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_1])
        stays_in_2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_2])        
        return stays_in_1 / cluster_volume_1 + stays_in_2 / cluster_volume_2 - self.rho


    def hierarchical_clustering_ShiMalik(self, K):
        networks = {}
        networks[0] = [self]

        for i in range(1,K):
            
            optimal_drhos = []
            optimal_cutoffs = []
            
            for nw in networks[i-1]:
                if nw.N<100: 
                    optimal_drhos.append(np.nan)
                    optimal_cutoffs.append(np.nan)
                    continue
        
                nw.compute_laplacian_spectrum()
                V_fiedler = nw.Lsym_eigenvectors[:,1]
                c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
                
                drhos = []
                for c in c_range:
                
                    indices_1 = np.argwhere(V_fiedler<=c)[:,0]
                    indices_2 = np.argwhere(V_fiedler>c)[:,0]
                    drhos.append(nw.drho_split(indices_1, indices_2))
                    
                drhos = np.array(drhos)
                plt.plot(c_range, drhos)
                plt.yscale('log')
                plt.grid(True)
                plt.title(r'$\Delta \rho_{global}$ for different cutoffs')
                plt.show()
                cutoff_opt = c_range[np.nanargmax(drhos)]
                print('Choosing as cutoff: ', str(cutoff_opt))
                
                optimal_drhos.append(np.nanmax(drhos))
                optimal_cutoffs.append(cutoff_opt)
            
            i_cluster = np.nanargmax(optimal_drhos)
            print('Splitting cluster ', i_cluster+1)
            cutoff_cluster = optimal_cutoffs[np.nanargmax(optimal_drhos)]
            nw_to_split = networks[i-1][i_cluster]
            V_fiedler = nw_to_split.Lsym_eigenvectors[:,1]
            indices_1 = np.argwhere(V_fiedler<=cutoff_cluster)[:,0]
            indices_2 = np.argwhere(V_fiedler>cutoff_cluster)[:,0]
            
            if len(indices_1)<len(indices_2):
                ind_ = indices_1.copy()
                indices_1= indices_2.copy()
                indices_2 = ind_
            
            adjacency_matrix_1 = nw_to_split.adjacency_matrix[indices_1, :][:, indices_1]
            adjacency_matrix_2 = nw_to_split.adjacency_matrix[indices_2, :][:, indices_2]
            cluster_indices_1 = nw_to_split.cluster_indices[indices_1]
            cluster_indices_2 = nw_to_split.cluster_indices[indices_2]
            cluster_volume_1 = nw_to_split.cluster_volume[indices_1]
            cluster_volume_2 = nw_to_split.cluster_volume[indices_2]
            
            
            cluster_label_1 = nw_to_split.cluster_label
            
            old_labels = [nw.cluster_label for nw in networks[i-1]]
            
            cluster_label_2 = np.max(old_labels)+1
            
            network_children = [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                            undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
            
            networks[i] = networks[i-1].copy()
            networks[i].pop(i_cluster)
            networks[i] += network_children #append in the back
        
        self.clustered_networks = networks


    def hierarchical_clustering_individual_rhos(self, rho_cutoff=0.99):
        networks = {}
        networks[0] = [self]
        
        for i in range(1,20):
            networks[i] = []
            for nw in networks[i-1]:
                if nw.rho>rho_cutoff and nw.N>100:
                    networks[i] += nw.split_according_to_v2(optimize = True)
                else:
                    networks[i] += [nw]
            
            if len(networks[i-1]) == len(networks[i]):
                print('No more splits executed')
                break;
        
        self.networks = networks


    def split_according_to_v2(self, optimize = False, promt=False):
        
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags(1./np.sqrt(d))
        L = sparse.identity(self.N) - D_sqrt_inv.dot(self.adjacency_matrix).dot(D_sqrt_inv)
        print('Symmetric normalized Laplacian computed. Computing spectrum.')
        
        w, v = sparse.linalg.eigsh(L, k=30, which = 'SM')
        v = D_sqrt_inv.dot(v)
        print('Spectrum computed')
        
        inds = np.argsort(w)
        w=w[inds]
        v=v[:,inds]
        plt.plot(w,'o')
        plt.title('Spectrum')
        plt.show()
        
        V_fiedler = v[:,1] #This transforms the eigenvectors
        
        plt.hist(V_fiedler, bins=50)
        plt.title('Histogram of splitting vector')
        plt.show()
        
        c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 300)[1:]
        
        ncuts = []        
        
        if optimize:
            for cutoff in c_range:
                indices_1 = np.argwhere(V_fiedler<=cutoff)[:,0]
                indices_2 = np.argwhere(V_fiedler>cutoff)[:,0]
                ncuts.append(self.ncut_split(indices_1, indices_2))
            
            ncuts = np.array(ncuts)
            plt.plot(c_range, ncuts)
            plt.yscale('log')
            plt.grid(True)
            plt.title('Ncuts for different cutoffs')
            plt.show()
            cutoff_opt = c_range[np.nanargmin(ncuts)]
            print('Choosing as cutoff: ', str(cutoff_opt))
            
            if promt:
                choice = input('Do you agree with this cutoff? (y/n)')
            else:
                choice = 'y'
            
            if choice!='y':
                cutoff_opt = float(input('Please specify the cutoff you want.'))
        else:
            print('Cutoff not optimized. Choosing c=0.')
            cutoff_opt=0
        
        self.c_range = c_range
        self.ncuts = ncuts
        
        indices_1 = np.argwhere(V_fiedler<=cutoff_opt)[:,0]
        indices_2 = np.argwhere(V_fiedler>cutoff_opt)[:,0]
        
        adjacency_matrix_1 = self.adjacency_matrix[indices_1, :][:, indices_1]
        adjacency_matrix_2 = self.adjacency_matrix[indices_2, :][:, indices_2]
        
        cluster_indices_1 = self.cluster_indices[indices_1]
        cluster_indices_2 = self.cluster_indices[indices_2]
        
        cluster_volume_1 = self.cluster_volume[indices_1]
        cluster_volume_2 = self.cluster_volume[indices_2]
        
        cluster_label_1 = self.cluster_label + '0'
        cluster_label_2 = self.cluster_label + '1'
        
        sub_clusters = [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
        
        if len(cluster_indices_1)<len(cluster_indices_2):
            sub_clusters = sub_clusters[::-1]
            
        return sub_clusters