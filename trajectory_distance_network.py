"""
Analysis with mean distance network:
    1. Construct matrix with M_ij = min(distances(i,j))
    2. Plot minimum distances
    3. Cluster with d_min as cutoff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from create_network import trajectory_data, undirected_network

drifter_trajectory_data = 'drifter_data_north_atlantic/uniformized_dataset_120days_1deg.npz'
matrix_file = "analysis_output/T_average_distances_uniformized_dataset_120days_1deg.npy"
d_min = 10 #cutoff, in km
    
    
def construct_T():
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval_days = 1)

f = plt.figure(constrained_layout=True, figsize = (10,5))
gs = f.add_gridspec(1, 1)
ax1 = f.add_subplot(gs[0,0])
indices = range(len(drifter_data.drifter_longitudes))
drifter_data.scatter_initial_position(ax1, indices)
plt.show()

drifter_data.compute_T()
np.save(matrix_file, drifter_data.T_mean_distances)

# construct_T()


def plot_mean_distances():   
    T = np.load(matrix_file)
    distances = T.flatten()
    distances = distances[~np.isnan(distances)]
    distances = distances[distances != 0]
    
    f = plt.figure(constrained_layout=True, figsize = (8,3.5))
    gs = f.add_gridspec(1, 2)
    
    h, x = np.histogram(distances, bins=600)
    x_range = x[:-1] + np.diff(x)/2.
    
    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Histogram', size=12)
    
    ax1.plot(x_range, h, 'k', linewidth = 3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Distance in km')
    ax1.set_ylabel('Drifter pair counts')
    ax1.grid(True)
    
    ax2 = f.add_subplot(gs[0, 1])
    ax2.set_title('Cumulative', size=12)
    ax2.plot(x_range[x_range<30], np.cumsum(h)[x_range<30], 'k', linewidth = 3)
    ax2.set_xscale('log')
    ax2.set_xlabel('Distance in km')
    ax2.set_ylabel('Drifter pair counts')
    ax2.grid(True)
    f.savefig('./figures/minimum_distances', dpi=300)


def cluster_meandistance_network():
    T = np.load(matrix_file)
    T = T + T.transpose()
    T = 1./T
    
    
    M = np.zeros(M_mindist.shape)
    M[M_mindist_symmetric<d_min]=1
    M = sparse.csr_matrix(M)
    
    A = undirected_network(M)
    A.connected_components()
    A_component1 = A.sub_networks[0]
    A_component1.compute_laplacian_spectrum()
    plt.plot(A_component1.Lsym_eigenvalues, 'o')

    drifter_data = trajectory_data.from_npz(drifter_trajectory_data)        
    
    for K in range(2,15):
            
        A_component1.cluster_kmeans(K=K)
        labels = A_component1.kmeans_labels
                
        fig, ax = plt.subplots(figsize = (10,10))        
        full_labels = np.zeros(A.adjacency_matrix.shape[0])
        full_labels[A_component1.cluster_indices] = labels +1
        full_labels = np.ma.masked_array(full_labels, full_labels==0)           
        drifter_data.scatter_initial_position_with_labels(ax, full_labels)
        ax.set_title(str(K))
        plt.show()

# cluster_chancepair_network()