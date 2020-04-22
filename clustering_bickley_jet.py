"""
Example for Bickley jet, 23 days
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from create_network import trajectory_data, undirected_network, bipartite_network
from sklearn.cluster import KMeans

r0 = 6371.

tau = 2073600
N = 125000
drifter_trajectory_data = "bickley_jet_trajectories_tau" + str(tau) + "_np_" + str(N) + '.npz'
domain_edges = (0., np.pi * r0, -5000, 5000) # full domain, to cover all points along any trajectory
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval_days = 1, 
                                        n_step=1, set_nans = False, domain_type = "bickley_jet_domain",
                                        domain_edges = domain_edges)
T = len(drifter_data.drifter_longitudes[0])


def complete_data_set():
    
    r = 10
    dx = np.max(np.diff(drifter_data.drifter_longitudes[:,0])) * r
    dy =  np.max(np.diff(drifter_data.drifter_latitudes[:,0])) * r
    d_deg = [dx, dy]
    drifter_data.compute_symbolic_sequences(d_deg = d_deg, 
                                        dt_days = 1)
    
    trajectory_lengths = np.array([len(np.unique(s)) for s in drifter_data.symbolic_sequence])
    f, ax = plt.subplots(figsize = (8,4))       
    drifter_data.scatter_position_with_labels_flat(ax, trajectory_lengths, colbar=False, t=0)
    plt.show()
    
    plt.hist(trajectory_lengths)
    plt.title('Lenghts of unique symbols, should be everywhere ' + str(T))
    
    #Define C-matrices
    C = []
    for i in range(T):
        C.append(drifter_data.compute_C(i))
    
    #Transfer operator for uniform distribution
    tranfer_operator_network = C[0].T.dot(C[T-1])
    F_transfer_operator = bipartite_network(tranfer_operator_network.tocsr())
    u_P, s_P, v_P = F_transfer_operator.stochastic_complement_laplacian_spectrum(K=25)
    
    #Particle closenes network
    mixing_network = C[0]
    
    for i in range(T):
        mixing_network = sparse.hstack((mixing_network,C[i]))
    
    F_mixing = bipartite_network(mixing_network.tocsr())
    u_M, s_M, v_M = F_mixing.projection_laplacian_spectrum(K=25)
    
    #path network
    path_network = C[0].copy()
    
    for i in range(1,T):
        path_network += C[i]
    
    F_path = bipartite_network(path_network.tocsr())
    u_path, s_path, v_path = F_path.projection_laplacian_spectrum(K=25)
    
    
    f = plt.figure(constrained_layout=True, figsize = (16,16))
    gs = f.add_gridspec(5, 3 )
    ax = f.add_subplot(gs[0, 0])
    ax.plot(range(25), s_P,'o', c = 'darkslategrey')
    ax.set_title('Transfer operator')
    plt.grid(True)
    
    ax = f.add_subplot(gs[0, 1])
    ax.plot(range(25),s_M,'o', c = 'darkslategrey')
    ax.set_title('Mixing network')
    plt.grid(True)
    
    ax = f.add_subplot(gs[0, 2])
    ax.plot(range(25),s_path,'o', c = 'darkslategrey')
    ax.set_title('Path network')
    plt.grid(True)
    
    Ks = [2,3,10,11]
    for i in range(4):
        K = Ks[i]
        ax = f.add_subplot(gs[i+1, 0])
        drifter_data.plot_discretized_distribution_flat(KMeans(n_clusters=K, random_state=0).fit(u_P[:,:K]).labels_, ax,
                                                       colbar=False, cmap="viridis")
        plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
        plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))
        
        
        ax = f.add_subplot(gs[i+1, 1])
        drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_M[:,:K]).labels_,
                                                       colbar=False, size=5)
        plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
        plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))
        
        
        ax = f.add_subplot(gs[i+1, 2])
        drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_,
                                                       colbar=False, size=5)
        plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
        plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))
    
    plt.savefig('figures/bickley_jet/bickley_jet_complete', dpi=300)


# complete_data_set()
    

# def incomplete_data_set():
      
r = 10
dx = np.max(np.diff(drifter_data.drifter_longitudes[:,0])) * r
dy =  np.max(np.diff(drifter_data.drifter_latitudes[:,0])) * r
d_deg = [dx, dy]

set_space = 1000
indices = np.random.randint(0, drifter_data.N, set_space)

print('Original data points:', str(T * drifter_data.N))

drifter_data.restrict_to_subset(indices)

for i in range(drifter_data.N):
    delete_time = np.random.randint(0,T)
    indices = np.random.randint(0, len(drifter_data.drifter_longitudes[0]), delete_time)
    drifter_data.drifter_longitudes[i][indices] = np.nan
    
print('Data points after restriction: ', str(np.sum([len(lo[~np.isnan(lo)]) for lo in drifter_data.drifter_longitudes])))

drifter_data.compute_symbolic_sequences(d_deg = d_deg, 
                                dt_days = 1)

trajectory_lengths = np.array([len(np.unique(s)) for s in drifter_data.symbolic_sequence])
f, ax = plt.subplots(figsize = (8,4))       
drifter_data.scatter_position_with_labels_flat(ax, trajectory_lengths, colbar=False, t=0)
plt.show()

plt.hist(trajectory_lengths)
plt.title('Lenghts of unique symbols, should be everywhere ' + str(T))

#Define C-matrices
C = []
for i in range(T):
    C.append(drifter_data.compute_C(i))

tranfer_operator_network = sparse.csr_matrix(np.zeros((C[0].shape[1], C[0].shape[1])))

for i in range(24):
    for j in range(i,24):
        tranfer_operator_network += C[i].T.dot(C[j])

F_transfer_operator = bipartite_network(tranfer_operator_network.tocsr())
u_P, s_P, v_P = F_transfer_operator.stochastic_complement_laplacian_spectrum(K=25)
i0_P = np.argwhere(s_P<0.999)[0][0]

path_network = C[0]

for i in range(T):
    path_network += C[i]

F_path = bipartite_network(path_network.tocsr())
u_path, s_path, v_path = F_path.stochastic_complement_laplacian_spectrum(K=10)
i0_path = np.argwhere(s_path<0.999)[0][0]

f = plt.figure(constrained_layout=True, figsize = (16,16))
gs = f.add_gridspec(5, 3 )
ax = f.add_subplot(gs[0, 0])
ax.plot(s_P,'o', c = 'darkslategrey')
ax.set_title('Transfer operator')
plt.grid(True)

ax = f.add_subplot(gs[0, 2])
ax.plot(s_path,'o', c = 'darkslategrey')
ax.set_title('Path network')
plt.grid(True)

Ks = [2,3,10,11]
for i in range(4):
    K = Ks[i]
    ax = f.add_subplot(gs[i+1, 0])
    # drifter_data.plot_discretized_distribution_flat(KMeans(n_clusters=K, random_state=0).fit(u_P[:,i0_P:i0_P+K]).labels_, ax,
    #                                                colbar=False, cmap="viridis")
    drifter_data.plot_discretized_distribution_flat(u_P[:,i0_P+K], ax,
                                                   colbar=False, cmap="viridis")
    plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
    plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))
    
    
    ax = f.add_subplot(gs[i+1, 2])
    drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_path[:,:K]).labels_,
                                                   colbar=False, size=5)
    plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
    plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))
    
    
# plt.savefig('figures/bickley_jet/bickley_jet_pathnetwork_incomplete', dpi=300)

# incomplete_data_set()



def complete_data_set_mixing():
        
    r = 50
    dx = np.max(np.diff(drifter_data.drifter_longitudes[:,0])) * r
    dy =  np.max(np.diff(drifter_data.drifter_latitudes[:,0])) * r
    d_deg = [dx, dy]
    
    drifter_data.compute_symbolic_sequences(d_deg = d_deg, 
                                    dt_days = 1)
    
    #Define C-matrices
    C = []
    for i in range(T):
        C.append(drifter_data.compute_C(i))
    
     #Particle closenes network
    mixing_network = C[0]
    
    for i in range(T):
        mixing_network = sparse.hstack((mixing_network,C[i]))
    
    F_mixing = bipartite_network(mixing_network.tocsr())
    u_M, s_M, v_M = F_mixing.projection_laplacian_spectrum(K=25)
    i0_M = np.argwhere(s_M<0.999)[0][0]
    
    f = plt.figure(constrained_layout=True, figsize = (8,4))
    gs = f.add_gridspec(2, 3)    
    ax = f.add_subplot(gs[0, 0])
    ax.plot(s_M[i0_M-1:],'o', c = 'darkslategrey')
    plt.grid(True)
    
    gs_s = [[0,1], [0,2], [1,0], [1,1], [1,2]]
    
    for K in range(2,7):
        
        ax = f.add_subplot(gs[gs_s[K-2][0], gs_s[K-2][1]])
        drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_M[:,i0_M:i0_M+K]).labels_,
                                                        colbar=False, size=20)
        plt.xticks(np.arange(0,2,0.5))
        plt.yticks(np.arange(0,1,0.5))
        plt.xticks(np.arange(0,30000,10000), np.arange(0,3,1))
        plt.yticks(np.arange(-2500,5000,2500), np.arange(-2.5,5,2.5))

# plt.savefig('figures/bickley_jet/mixing_r_' + str(r), dpi=300)

# incomplete_data_set()



