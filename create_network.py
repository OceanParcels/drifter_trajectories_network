# -*- coding: utf-8 -*-
"""
Classes to process drifter data into networks / datasets for unsupervised learning.
IMPORTANT NOTES:
    - time interval of 6h is assumed

"""
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from scipy.sparse import coo_matrix
from datetime import datetime, timedelta
from scipy import sparse
import scipy.sparse
import networkx as nx
from scipy.sparse.linalg import eigs

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
    """
    Class to
    - handle drifter data and compute the networkd / datasets used for unsupervised learning
    - plot a field over North Atlantic Ocean
    """
    
    def __init__(self, drifter_longitudes=[], drifter_latitudes=[], 
                 drifter_time=[], drifter_SST = [], drifter_speed = [], 
                 drifter_id=[], data_description = "no description",
                 set_nans = True):
        """
        drifter_longitudes: format is len-N list of numpy arrays, each array containing the data.
        Similar for drifter_latitudes, drifter_time (datetime.timestamp), drifter_SST, drifter_speed.
        drifter_time is ASSUMED TO BE in 6h intervals
        drifter_id: len-N list of ID's
        data_description: option to specify what kind of data it is
        set_nans: Some drifters have erroneous data. If True, this data is set to nan
        """

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
    
        self.discretized = False #No partition is defined in the beginning
    
    
    def set_nans(self):
        """
        The drifter data contains some empty data (where lon / lat are very large)
        """
        print('Setting erroneous data points to nan')
        
        n_data_errors = 0
        
        for i in range(self.N):
            n_data_errors += len(self.drifter_longitudes[i][self.drifter_longitudes[i]>180])
            self.drifter_latitudes[i][self.drifter_longitudes[i]>180]=np.nan
            self.drifter_longitudes[i][self.drifter_longitudes[i]>180]=np.nan
            
        print("found " + str(n_data_errors) + " nans")            
        self.nans_set = True
    
    
    def start_end_times(self):
        self.start_times = [timedelta(seconds = self.drifter_time[i][0]) + datetime(1900, 1, 1, 0, 0) for i in range(len(self.drifter_time))]
        self.end_times = [timedelta(seconds = self.drifter_time[i][-1]) + datetime(1900, 1, 1, 0, 0) for i in range(len(self.drifter_time))]
        
        
    
    
    def distance_on_sphere(self, i1, i2, trajectory_type):
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
        
    
    def compute_M(self, min_dist):
        """
        Function to compute NxN network containing the minimum distance (simultaneous time)
        of two drifters
        """
        
        if not self.nans_set:
            self.set_nans()
        
        A = np.empty((self.N, self.N))
        
        for i in range(self.N):
            if i%50 ==0: print(str(i) + " / " + str(self.N))
            for j in range(i+1, self.N):
                A[i,j] = np.nanmin(self.distance_on_sphere(i,j))
    
        self.M_mindist = A + A.transpose()
        self.M = np.zeros(A.shape)
        self.M[self.M_mindist<=min_dist] = 1
    
    
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
        """
        Function to compute the matrix index from a given (lon, lat) coordinate
        """
        (lon, lat) = coord_2D
        if np.isnan(lon) or np.isnan(lat):
            return np.nan
        else:
            index_2D    = int((lat - self.domain.minlat)//self.d_deg * self.n_lons + (lon - self.domain.minlon)//self.d_deg)
            return index_2D


    def compute_symbolic_sequences(self, d_deg=0.5, dt_days=5):
        """
        Function to compute, for each drifter, the symbolic sequence given the 
        cell size of d_deg and time interval of dt_days. See respective section
        in the paper for more detailed explanation.
        """
        
        self.set_discretizing_values(d_deg)
        self.dt_days = dt_days
        dt_indices = int(dt_days * 4) #as data is every 6 hours
        
        self.symbolic_sequence = []
        
        for k in range(dt_indices):
            self.symbolic_sequence += [np.array([self.coords_to_matrixindex2D((lo,la)) for lo, la in 
                                      zip(self.drifter_longitudes[i][k::dt_indices],
                                          self.drifter_latitudes[i][k::dt_indices])]) for i in range(self.N)]
        
    
    def compute_P_and_C(self, matrix_type='killed_Markov'):
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
        # from sklearn.preprocessing import normalize
        # P = normalize(W, norm='l1', axis=1)
        P =  (W.transpose().dot(D_inv)).transpose()
        
        if matrix_type == 'Markov':
            s = np.array(sparse.csr_matrix.sum(P, axis=1))[:,0]
            assert(np.all(s[s>0]>1.-1e-9))
                    
        self.initial_indices = initial_indices
        self.final_indices = final_indices
        self.W = W
        self.d = d
        self.P = P
        self.C = P.copy()
        self.C.data[:] = 1
        
        
    def compute_T(self):
        
        symbolic_sequence = self.symbolic_sequence
        rows = []
        cols = []
        vals = []
        
        for i in range(len(symbolic_sequence)):
            if i%50 ==0: print(str(i) + " / " +str(len(symbolic_sequence)))
            s1 = set(symbolic_sequence[i])
            for j in range(i, len(symbolic_sequence)):
                s2 = set(symbolic_sequence[j])
                v = len(set.intersection(s1, s2))
                if v>0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)
         
        self.C_counts = coo_matrix((vals, (rows, cols)), shape=(len(symbolic_sequence),len(symbolic_sequence))).tocsr()
        self.C = self.C_counts.copy()
        self.C.data[:]=1


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


    def plot_discretized_distribution(self, v, ax, cmap = None, land = False, 
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
        
        m = Basemap(projection='mill',llcrnrlat=0.,urcrnrlat=85., 
                    llcrnrlon=-100,urcrnrlon=40,resolution='c')

        m.drawparallels([20,40,60,80], labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians([-70,-40,-10,20,50], labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()

        if land: m.fillcontinents(color='dimgray')
        xs, ys = m(self.domain.lon_bins_2d, self.domain.lat_bins_2d)
        if logarithmic:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap,  norm=colors.LogNorm())
        else:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap) # , vmin=-20, vmax=1)
        # ax.set_title(title, size=20, y=1.01)

        if colbar: 
            if cbar_orientation == 'horizontal':
                cbar = plt.colorbar(p, shrink=.6, aspect=10, orientation="horizontal")
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=40)
            cbar.ax.tick_params(labelsize=9) 
            if cbartitle is not None:
                cbar.set_label(cbartitle, size=9)

        # return p
        

class undirected_network(object):
    
    def __init__(self, adjacency_matrix, cluster_indices, cluster_volume, 
                 cluster_label = '0'):
        """
        adjacency_matrix: format sparse.csr_matrix. If it is not symmetric it is symmetrized.
        region_indices: indices corresponding to network domain.
        cluster_volume: vector of volume of the nodes inside the cluster.
        """
        self.adjacency_matrix = adjacency_matrix
        self.cluster_indices = cluster_indices
        self.cluster_volume = cluster_volume
        self.N = len(cluster_indices)
        self.cluster_label = cluster_label
        self.rho = np.sum(self.adjacency_matrix)/np.sum(self.cluster_volume)
        assert(len(cluster_indices) == self.adjacency_matrix.shape[0])
        assert(len(cluster_volume) == len(cluster_indices))
        print('Construct undirected network.')
    
    def __del__(self):
        print('Adjacency matrix deleted')

    @classmethod
    def from_sparse_npz(cls, filename):
        """
        Load from sparse matrix
        """
        A = scipy.sparse.load_npz(filename)
        return cls(A)

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
        
        for c in components_sorted:
            inds = list(c)
            sub_adjacency_matrix = self.adjacency_matrix[inds, :][:, inds]
            sub_cluster_indices = self.cluster_indices[inds]
            sub_cluster_volume = self.cluster_volume[inds]
            sub_cluster_label = self.cluster_label + '_'
            sub_networks.append(undirected_network(sub_adjacency_matrix, sub_cluster_indices, 
                                                   sub_cluster_volume, sub_cluster_label))
        
        self.sub_networks = sub_networks
        

    def compute_laplacian_spectrum(self, K=20):
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags(1./np.sqrt(d))
        L = sparse.identity(self.N) - (D_sqrt_inv.dot(self.adjacency_matrix)).dot(D_sqrt_inv)
        print('Computing spectrum')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        self.eigenvalues = w
        self.eigenvectors = D_sqrt_inv.dot(v)


    def ncut_split(self, indices_1, indices_2):
        a1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_2])
        a2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_1])
        
        s1 = np.sum(self.adjacency_matrix[indices_1,:])
        s2 = np.sum(self.adjacency_matrix[indices_2,:])
        
        return a1/s1 + a2/s2


    def drho_split(self, indices_1, indices_2):
        cluster_volume_1 = np.sum(self.cluster_volume[indices_1])
        cluster_volume_2 = np.sum(self.cluster_volume[indices_2])
        
        stays_in_1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_1])
        stays_in_2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_2])
        
        return stays_in_1 / cluster_volume_1 + stays_in_2 / cluster_volume_2 - self.rho


    def split_according_to_vn(self, optimize = False, n=2):
        
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
        
        V_fiedler = v[:,n-1] #This transforms the eigenvectors
        
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
            plt.hlines(w[n-1], np.min(ncuts), np.max(ncuts), colors = 'r', linestyles = '--')
            plt.grid(True)
            plt.title('Ncuts for different cutoffs')
            plt.show()
            cutoff_opt = c_range[np.nanargmin(ncuts)]
            print('Choosing as cutoff: ', str(cutoff_opt))
            
            choice = input('Do you agree with this cutoff? (y/n)')
            
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


def symbolic_paths_matrix():
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    description = "trajectory based network, not time info"
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID, 
                                    data_description = description)
    drifter_data.compute_symbolic_sequences(d_deg = 0.1, dt_days = 0.25)
    
    sequence = drifter_data.symbolic_sequence
    
    rows = []
    cols = []
    vals = []
    
    for i in range(len(sequence)):
        if i%50 ==0: print(str(i) + " / " +str(len(sequence)))
        s1 = set(sequence[i])
        for j in range(i, len(sequence)):
            s2 = set(sequence[j])
            v = len(set.intersection(s1, s2))
            if v>0:
                rows.append(i)
                cols.append(j)
                vals.append(v)
     
    W = coo_matrix((vals, (rows, cols)), shape=(len(sequence),len(sequence))).tocsr()
    sparse.save_npz('trajectory_network', W)
 

def compute_drifter_transitionmatrix():
                    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    dt_days=60
    d_deg = 1.0
    print('dt_days: ', dt_days)
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID)
    drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = dt_days)
    
    print("computing transition matrix")
    drifter_data.compute_P_and_C(matrix_type = 'Markov')
    sparse.save_npz('analysis_output/transition_matrix_markov_ddeg1_ddays_60', drifter_data.P)
    np.save('analysis_output/initial_distribution_markov_ddeg1', drifter_data.d)
    # sparse.save_npz('analysis_output/count_matrix_markov_ddeg05_ddays', drifter_data.W)

# compute_drifter_transitionmatrix()