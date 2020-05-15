"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Check number of incorrectly assigned labels with incomplete data, compared to full data case
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data, bipartite_network
from sklearn.cluster import KMeans

T_G = 201 #time used to construct matrices. Set to 1 to constrain to double gyre period.

#Complete data set
dx, dy = 0.04,0.04
domain_edges = (0,2.,0,1.) #we set the domain edges such  that for both (complete and incomplete cases) the partition is the same

#load trajectory data
drifter_trajectory_data = "double_gyre_nonautonomous_trajectories.npz"
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                    n_step=1, domain_type = "double_gyre_domain",
                                    domain_edges = domain_edges)

T = len(drifter_data.drifter_longitudes[0])
bin_size = [dx, dy]

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
u_complete, s_complete, v_complete = F.projection_laplacian_spectrum(K=10)


import itertools
incorrect_particles = {}

ds = np.arange(0.1, 1., 0.1)

k=2
labels_complete_full =  KMeans(n_clusters=k, random_state=0).fit(u_complete[:,:k]).labels_
incorrect_particles[k]={}

for d in range(len(ds)):
    print(d)
    incorrect_particles[k][d] = []
    for _ in range(100):
        #Incomplete data set
        ratio_delete = 0.8
        bin_size = [ds[d], ds[d]]
        n_particles=500
        drifter_data_incomplete = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                        n_step=1, domain_type = "double_gyre_domain",
                                        domain_edges = domain_edges)
        indices = np.random.randint(0, drifter_data_incomplete.N, n_particles)
        drifter_data_incomplete.restrict_to_subset(indices) #reduce particle set
        
        delete_matrix = np.random.uniform(0,1, (drifter_data_incomplete.N,T))
        delete_matrix[delete_matrix<ratio_delete]=0
        delete_matrix[delete_matrix>ratio_delete]=1
        for i in range(drifter_data_incomplete.N):
            drifter_data_incomplete.drifter_longitudes[i] = np.array([drifter_data_incomplete.drifter_longitudes[i][j] if delete_matrix[i,j]==1 else np.nan for j in range(T)])
            drifter_data_incomplete.drifter_latitudes[i] = np.array([drifter_data_incomplete.drifter_latitudes[i][j] if delete_matrix[i,j]==1 else np.nan for j in range(T)])
        
        drifter_data_incomplete.compute_symbolic_itineraries(bin_size = bin_size, dt = 1)
        
        C_incomplete = []
        for i in range(T): C_incomplete.append(drifter_data_incomplete.compute_C(i))
        
        #define G(tau)
        G_incomplete = C_incomplete[0]
        for i in range(1, T_G): G_incomplete += C_incomplete[i]
        
        F_incomplete = bipartite_network(G_incomplete.tocsr())
        u_incomplete, s_incomplete, v_incomplete = F_incomplete.projection_laplacian_spectrum(K=k)
       
        permutations = list(itertools.permutations(list(range(k))))
        labels_incomplete = KMeans(n_clusters=k, random_state=0).fit(u_incomplete[:,0:k]).labels_
        labels_complete = labels_complete_full[indices]
        
        incorrects = []
        for p in permutations:
            new_labels_incomplete = np.ones(len(indices)) * (-1)
            
            for i, j in zip(range(k),p):
                new_labels_incomplete[labels_incomplete==i] = j
        
            incorrect=0
            for i in range(len(indices)):
                if new_labels_incomplete[i] != labels_complete[i]:
                    incorrect+=1
            
            incorrects.append(incorrect)
        
        incorrect_particles[k][d].append(np.min(incorrects))

colors = ['','','darkslategrey', 'maroon', 'midnightblue']
y = np.array([np.mean(incorrect_particles[k][d]) for d in range(len(ds))])
y_std = np.array([np.std(incorrect_particles[k][d]) for d in range(len(ds))])
y_max = np.array([np.max(incorrect_particles[k][d]) for d in range(len(ds))])
y_min = np.array([np.min(incorrect_particles[k][d]) for d in range(len(ds))])
y_min2 = y - y_std
y_max2 = y + y_std

plt.figure(figsize=(5,3))
plt.fill_between(ds, y_min, y_max, alpha=0.2, color = 'darkslategrey', label = "min / max")
plt.fill_between(ds, y_min2, y_max2, alpha=0.2, color = 'maroon', label = r"y $\pm$ std")

plt.plot(ds, y, marker='o', linestyle = '--', color = "k", markersize=4, label = 'mean')
# plt.errorbar(ds, y, yerr=y_std, marker='o', linestyle='', label=str(k))
plt.xlabel(r'$\Delta x$ $(= \Delta y)$')
plt.yticks(np.arange(0,150,20), np.arange(0,150,20)/n_particles*100)
plt.ylim([0,100])
plt.ylabel('% wrongly identified particles (K=2)')
plt.grid(True)    
plt.legend()
plt.tight_layout()
plt.savefig('figures/dg_sensitivity_incomplete_data', dpi=300)
