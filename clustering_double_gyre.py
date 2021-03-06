"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Clustering of (non-)autonomous double gyre for complete and incomplete data sets
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data, bipartite_network
from sklearn.cluster import KMeans
import itertools

ratio_delete = 0 #Minimum amount of time info deleted in each of the remaining 500 trajectories.
                #only needed for incomplete cases, where this varable is re-defined
T_G = 201

"""
Here are the options for the different figures. Please comment the blocks you don't need.
"""

"""
Non-autonomous case
"""
drifter_trajectory_data = "double_gyre_nonautonomous_trajectories.npz"

# #Complete data set (fig 3.)
# incomplete = False
# filename = './figures/dg_complete_dxdy_004_T' + str(T_G)
# K_clusters=15
# dx, dy = 0.04, 0.04
# figlines = 2

#Complete data set, shorter time (fig 3.)
incomplete = False
T_G = 101
filename = './figures/dg_complete_dxdy_004_T' + str(T_G)
K_clusters=15
dx, dy = 0.04, 0.04
figlines = 2

# Complete data set, coarse binning (fig. 4)
# incomplete = False
# filename = './figures/dg_complete_dxdy_066_0333_' + str(T_G)
# K_clusters=5
# dx, dy = 2/3., 1./3.
# figlines = 1
    
# Incomplete data set (fig. 5)
# incomplete = True
# filename = './figures/dg_incomplete_dxdy_004_T' + str(T_G)
# K_clusters=10
# dx, dy = 0.04, 0.04
# dx_incomplete, dy_incomplete = .4, .4
# ratio_delete = 0.8
# figlines = 1

"""
Autonomous case (figs. B1, B2)
"""

# drifter_trajectory_data = "double_gyre_autonomous_trajectories.npz"
 
# #Autonomous case, optimal partition (fig. B1)
# incomplete = False
# filename = './figures/dg_autonomous_complete_dxdy_01_T' + str(T_G)
# K_clusters=16
# dx, dy = 0.1, 0.1
# figlines = 1

# #Autonomous case, non-optimal partition (fig. B2)
# incomplete = False
# filename = './figures/dg_autonomous_complete_dxdy_015_T' + str(T_G)
# K_clusters=16
# dx, dy = 0.15, 0.15
# figlines = 1



#load trajectory data
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                    n_step=1, domain_type = "double_gyre_domain")
T = len(drifter_data.drifter_longitudes[0])

bin_size = [dx, dy]

if incomplete == True: #reduce particle set
    drifter_data_incomplete = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                    n_step=1, domain_type = "double_gyre_domain")
    set_space = 500 #number of particles
    indices = np.random.randint(0, drifter_data_incomplete.N, set_space)
    print('Original data points:', str(T * drifter_data_incomplete.N))
    drifter_data_incomplete.restrict_to_subset(indices) #reduce particle set

    delete_matrix = np.random.uniform(0,1,(drifter_data_incomplete.N,T))
    delete_matrix[delete_matrix<ratio_delete]=0
    delete_matrix[delete_matrix>ratio_delete]=1
    
    for i in range(drifter_data_incomplete.N):
        drifter_data_incomplete.drifter_longitudes[i] = np.array([drifter_data_incomplete.drifter_longitudes[i][j] if delete_matrix[i,j]==1 else np.nan for j in range(T)])
        drifter_data_incomplete.drifter_latitudes[i] = np.array([drifter_data_incomplete.drifter_latitudes[i][j] if delete_matrix[i,j]==1 else np.nan for j in range(T)])
    
    print('Data points after restriction: ', str(np.sum([len(lo[~np.isnan(lo)]) for lo in drifter_data_incomplete.drifter_longitudes])))


drifter_data.compute_symbolic_itineraries(bin_size = bin_size, dt = 1)
total_symbols = []
for s in drifter_data.symbolic_sequence: total_symbols += list(s[:])
print('unique symbols: ', len(np.unique(total_symbols)))
print(np.unique(total_symbols))
#define list of C(t) matrices
C = []
for i in range(T): C.append(drifter_data.compute_C(i))

#define G(tau)
G = C[0]
for i in range(1, T_G): G += C[i]

F = bipartite_network(G.tocsr())
u, s, v = F.projection_laplacian_spectrum(K=K_clusters)
    
if incomplete:
    drifter_data_incomplete.compute_symbolic_itineraries(bin_size = [dx_incomplete, dy_incomplete], dt = 1)
    
    C_incomplete = []
    for i in range(T): C_incomplete.append(drifter_data_incomplete.compute_C(i))
    
    #define G(tau)
    G_incomplete = C_incomplete[0]
    for i in range(1, T_G): G_incomplete += C_incomplete[i]
    
    F_incomplete = bipartite_network(G_incomplete.tocsr())
    u_incomplete, s_incomplete, v_incomplete = F_incomplete.projection_laplacian_spectrum(K=K_clusters)

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']


#Figure
f = plt.figure(constrained_layout=True, figsize = (10,figlines * 2.5))
gs = f.add_gridspec(figlines, 4)
ax = f.add_subplot(gs[0, 0])
if incomplete: ax.plot(range(K_clusters), s_incomplete,'o', c = 'darkslategrey')
else: ax.plot(range(K_clusters), s,'o', c = 'darkslategrey')
ax.set_title(r'(a) singular values of $R$')
plt.grid(True)

for K in range(2,np.min([5,K_clusters+1])):
    ax = f.add_subplot(gs[0, K-1])
        
    drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u[:,:K]).labels_,
                                                       cbar=False, size=0.15, t=0, cmap = 'Paired')
    
    if incomplete:
        drifter_data_incomplete.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_incomplete[:,:K]).labels_,
                                                       cbar=False, size=4., t=0, cmap = 'Set1')
    
    
    ax.set_title(panel_labels[K-1] + ' K = ' + str(K))
    
    if K==2: plt.yticks(np.arange(0,1.5,0.5))
    else: plt.yticks([])
    
    if figlines==1: plt.xticks(np.arange(0,2.5,1))
    else: plt.xticks([])
        

for K in range(5,np.min([figlines*4+1,K_clusters+1])):
    ax = f.add_subplot(gs[1, K-5])

    drifter_data.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u[:,:K]).labels_,
                                                       cbar=False, size=0.15, t=0, cmap = 'Paired')
    
    if incomplete:
        drifter_data_incomplete.scatter_position_with_labels_flat(ax, KMeans(n_clusters=K, random_state=0).fit(u_incomplete[:,:K]).labels_,
                                                       cbar=False, size=4., t=0, cmap = 'Set1')

    ax.set_title(panel_labels[K-1] + ' K = ' + str(K))
    plt.xticks(np.arange(0,2.5,1))
    if K == 5: plt.yticks(np.arange(0,1.5,0.5))
    else: plt.yticks([])

f.savefig(filename, dpi=300)


if incomplete:
    permutations = list(itertools.permutations(list(range(2))))
    labels_incomplete = KMeans(n_clusters=2, random_state=0).fit(u_incomplete[:,:2]).labels_
    labels_complete = KMeans(n_clusters=2, random_state=0).fit(u[:,:2]).labels_[indices]
    
    incorrects = []
    for p in permutations:
        new_labels_incomplete = np.ones(len(indices)) * (-1)
        
        for i, j in zip(range(2),p):
            new_labels_incomplete[labels_incomplete==i] = j
    
        incorrect=0
        for i in range(len(indices)):
            if new_labels_incomplete[i] != labels_complete[i]:
                incorrect+=1
        
        incorrects.append(incorrect)

    print('Incorrectly assigned particles: ', np.min(incorrects))
