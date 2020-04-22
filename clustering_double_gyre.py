"""
Application to the double gyre

"""

import numpy as np
import matplotlib.pyplot as plt
from create_network import trajectory_data, bipartite_network
from sklearn.cluster import KMeans

#parameters
r = 4
plot_t = 0 #only for particle network
incomplete = True
particle_network = False
bin_network = True
K_clusters=15
ratio_delete = 0.98

#load trajectory data
drifter_trajectory_data = "double_gyre_trajectories_np_20000_tau_20_dt_0.1.npz"
domain_edges = (0., 2., 0. , 1.)
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval_days = 1, 
                                    n_step=1, set_nans = False, domain_type = "double_gyre_domain",
                                    domain_edges = domain_edges)
T = len(drifter_data.drifter_longitudes[0])

dx = np.max([(drifter_data.drifter_longitudes[i+1][0] - drifter_data.drifter_longitudes[i][0]) for i in range(drifter_data.N-1)]) * r
dy = np.max([(drifter_data.drifter_latitudes[i+1][0] - drifter_data.drifter_latitudes[i][0]) for i in range(drifter_data.N-1)]) * r

dx, dy = 0.25, 0.25

d_deg = [dx, dy]

if incomplete == True: #reduce particle set
    set_space = 500 #number of particles
    indices = np.random.randint(0, drifter_data.N, set_space)
    print('Original data points:', str(T * drifter_data.N))
    drifter_data.restrict_to_subset(indices) #reduce particle set
    
    for i in range(drifter_data.N):
        delete_time = np.random.randint(int(T* ratio_delete),T)
        I = np.array(range(T))
        np.random.shuffle(I)
        indices = I[:delete_time]
        drifter_data.drifter_longitudes[i][indices] = np.nan
    
    print('Share of data points after restriction: ', str(np.sum([len(lo[~np.isnan(lo)]) for lo in drifter_data.drifter_longitudes])))

drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = 1)

#define list of C(t) matrices
C = []
for i in range(T): C.append(drifter_data.compute_C(i))

#define G(tau)
G = C[0]
for i in range(T): G += C[i]

if particle_network:

    #Spectrum of normalized Laplacian
    F_path = bipartite_network(G.tocsr())
    u_path, s_path, v_path = F_path.projection_laplacian_spectrum(K=K_clusters)
    
    #Figure
    f = plt.figure(constrained_layout=True, figsize = (10,5))
    gs = f.add_gridspec(2, 4)
    ax = f.add_subplot(gs[0, 0])
    ax.plot(range(K_clusters),s_path,'o', c = 'darkslategrey')
    ax.set_title('Singular values of M_p')
    plt.grid(True)
    
    for K in range(2,5):
        ax = f.add_subplot(gs[0, K-1])
        drifter_data.scatter_position_with_labels_flat(KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_, ax,
                                                       colbar=False, size=15, t=plot_t, cmap = 'Paired')
        ax.set_title('K = ' + str(K))
        if K==2: plt.yticks(np.arange(0,1.5,0.5))
        else: plt.yticks([])
        plt.xticks([])
    
    
    for K in range(5,9):
        ax = f.add_subplot(gs[1, K-5])
        drifter_data.scatter_position_with_labels_flat(KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_, ax,
                                                        colbar=False, size=15, t=plot_t, cmap = 'Paired')
        ax.set_title('K = ' + str(K))
        plt.xticks(np.arange(0,2.5,1))
        if K == 5: plt.yticks(np.arange(0,1.5,0.5))
        else: plt.yticks([])
    
    if incomplete == True:
        f.savefig('./figures/double_gyre/double_gyre_incomplete_r_' + str(r) + '_plot_t' + str(plot_t), dpi=300)
    else:
        f.savefig('./figures/double_gyre/double_gyre_complete_r_' + str(r) + 'plot_t' + str(plot_t), dpi=300)


if bin_network:
    #Spectrum of normalized Laplacian
    F_path = bipartite_network(G.transpose().tocsr())
    u_path, s_path, v_path = F_path.stochastic_complement_laplacian_spectrum(K=K_clusters)
    
    #Figure
    f = plt.figure(constrained_layout=True, figsize = (10,5))
    gs = f.add_gridspec(2, 4)
    ax = f.add_subplot(gs[0, 0])
    ax.plot(range(K_clusters),s_path,'o', c = 'darkslategrey')
    ax.set_title('Singular values of M_b')
    plt.grid(True)
    
    for K in range(2,5):
        ax = f.add_subplot(gs[0, K-1])
        drifter_data.plot_discretized_distribution_flat(KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_, ax,
                                                       colbar=False, cmap='Paired')
        ax.set_title('K = ' + str(K))
        if K==2: plt.yticks(np.arange(0,1.5,0.5))
        else: plt.yticks([])
        plt.xticks([])
        
    
    for K in range(5,9):
        ax = f.add_subplot(gs[1, K-5])
        drifter_data.plot_discretized_distribution_flat(KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_, ax,
                                                        colbar=False, cmap='Paired')
        ax.set_title('K = ' + str(K))
        plt.xticks(np.arange(0,2.5,1))
        if K == 5: plt.yticks(np.arange(0,1.5,0.5))
        else: plt.yticks([])
    
    if incomplete == True:
        f.savefig('./figures/double_gyre/double_gyre_bins_incomplete_r' + str(r) + '_plot_t' + str(plot_t), dpi=300)
    else:
        f.savefig('./figures/double_gyre/double_gyre_bins_complete_r_' + str(r) + '_plot_t' + str(plot_t), dpi=300)
