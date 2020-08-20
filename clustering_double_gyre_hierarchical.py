"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Clustering of (non-)autonomous double gyre for hierarchical clustering method
for different values of bin size.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data, bipartite_network, undirected_network, construct_dendrogram
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.colors

drifter_trajectory_data = "double_gyre_nonautonomous_trajectories.npz"

T_G = 201
K, L = 4, 4

# for DX in [0.04, 0.1, 0.2]:
DX = 0.1
dx, dy = DX, DX

#load trajectory data
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                    n_step=1, domain_type = "double_gyre_domain")
T = len(drifter_data.drifter_longitudes[0])

# indices = range(0,drifter_data.N,4)
# drifter_data.restrict_to_subset(indices)

bin_size = [dx, dy]

drifter_data.compute_symbolic_itineraries(bin_size = bin_size, dt = 1)

#define list of C(t) matrices
C = []
for i in range(T): C.append(drifter_data.compute_C(i))

#define G(tau)
G = C[0]
for i in range(1, T_G): G += C[i]

# #Set up networks and hierarchical clustering according to Shi&Malik 2000
B = bipartite_network(G)
A = undirected_network(B.projection_adjacency_matrix(space = 'X'))    

w, v = A.compute_laplacian_spectrum(K=20, plot=False)
drhos, c_range, cutoff_opt = A.hierarchical_clustering_ShiMalik(K, plots=True)
np.savez('dg_hierarchical_clustering_firstsplit' + str(dx), drhos=drhos, c_range=c_range)
networks = A.clustered_networks

L=4
Z = construct_dendrogram([networks[i] for i in range(L)])

f = plt.figure(constrained_layout=True, figsize = (10,3))
gs = f.add_gridspec(1, 3)

ax = f.add_subplot(gs[0, 0])
ax.plot(c_range, 1-drhos, 'o-', color = 'darkslategrey', markersize=3, linewidth=2)
ax.grid(True)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax.set_ylabel('NCut')
ax.set_xlabel('c')
ax.axvline(cutoff_opt, linestyle = '--', color = 'darkred', linewidth=2)
ax.set_title('(a) NCut for the first split')


ax = f.add_subplot(gs[0, 2])
matplotlib.rcParams['lines.linewidth'] = 3
cmap = matplotlib.cm.get_cmap('Paired')
colors = cmap(range(4))
# colors = np.array(['r','g','k','yellow'])

labels = ['A', 'B', 'C', 'D'] #, 'E', 'F', 'G', 'H',]
hierarchy.dendrogram(Z, color_threshold=0.2, distance_sort='ascending', labels=labels, 
                      link_color_func=lambda k: 'k')

ncut = []
for i in range(len(networks)):
    r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
    ncut.append(len(networks[i])-r)
ncut=np.array(ncut)[:L]

plt.yticks(np.arange(np.max(ncut)+.2, 0, -0.5)[::-1], np.max(ncut)+.2-np.arange(np.max(ncut)+.2, 0, -0.5)[::-1])
ax.set_ylabel('Ncut')
ax.set_title('(c) cluster hierarchy')
    
ax = f.add_subplot(gs[0, 1])
ax.set_title('(b) particle labels at initial time')
bounds = np.arange(-0.5,L+0.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))

cmap = matplotlib.colors.ListedColormap(colors)   
field_plot = np.ones(G.shape[0]) *(-10000)
for k in range(L): 
    field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label

field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
drifter_data.scatter_position_with_labels_flat(ax, field_plot, cmap=cmap, norm=norm,
                                                  cbar=False, size=.5, t=0)

dx_labelparams = {0.04: 
                  {'A_loc': (0.3,0.8), 'A_col_idx': 0, 
                    'B_loc': (0.7,0.2), 'B_col_idx': 1,
                    'C_loc': (0.2,0.45), 'C_col_idx': 2,
                    'D_loc': (0.75,0.5), 'D_col_idx': 3},
                  0.1: 
                      {'A_loc': (0.45,0.8), 'A_col_idx': 0, 
                    'B_loc': (0.35,0.2), 'B_col_idx': 1,
                    'C_loc': (0.75,0.5), 'C_col_idx': 2,
                    'D_loc': (0.2,0.45), 'D_col_idx': 3},
                   0.2: 
                      {'A_loc': (0.3,0.8), 'A_col_idx': 0, 
                    'B_loc': (0.6,0.2), 'B_col_idx': 1,
                    'C_loc': (0.2,0.45), 'C_col_idx': 2,
                    'D_loc': (0.7,0.45), 'D_col_idx': 3}
                  }

dx_labels = dx_labelparams[dx]

t =plt.annotate('A', dx_labels['A_loc'], xycoords='axes fraction', size=20, color=cmap(dx_labels['A_col_idx']))
t.set_bbox(dict(facecolor='k', alpha=0.5))
    
t =plt.annotate('B', dx_labels['B_loc'], xycoords='axes fraction', size=20, color=cmap(dx_labels['B_col_idx']))
t.set_bbox(dict(facecolor='k', alpha=0.5))

t =plt.annotate('C', dx_labels['C_loc'], xycoords='axes fraction', size=20, color=cmap(dx_labels['C_col_idx']))
t.set_bbox(dict(facecolor='k', alpha=0.5))

t =plt.annotate('D', dx_labels['D_loc'], xycoords='axes fraction', size=20, color=cmap(dx_labels['D_col_idx']))
t.set_bbox(dict(facecolor='k', alpha=0.5))

f.savefig('./figures/dg_hierarchical_' + str(dx).replace('.',''), dpi=300)