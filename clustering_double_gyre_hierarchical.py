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
from particle_and_network_classes import trajectory_data, bipartite_network, undirected_network, construct_dendrogram
from sklearn.cluster import KMeans
import itertools
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.colors

drifter_trajectory_data = "double_gyre_nonautonomous_trajectories.npz"

#Complete data set (fig 3.)
incomplete = False
T_G = 201
filename = './figures/dg_hierarchical_dxdy_004_T' + str(T_G)
K_clusters=15
dx, dy = 0.04, 0.04
figlines = 2
K = 10
L = 8

#load trajectory data
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, time_interval = 1, 
                                    n_step=1, domain_type = "double_gyre_domain")
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

#Set up networks and hierarchical clustering according to Shi&Malik 2000
B = bipartite_network(G)
A = undirected_network(B.projection_adjacency_matrix(space = 'X'))    

# for the double-gyre, the network is connected (no need to check for largest connected componen)
w, v = A.compute_laplacian_spectrum(K=40)
A.hierarchical_clustering_ShiMalik(K)
networks = A.clustered_networks

Z = construct_dendrogram([networks[i] for i in range(L)])

f = plt.figure(constrained_layout=True, figsize = (10,8))
gs = f.add_gridspec(2, 4)
ax = f.add_subplot(gs[1, 1:3])
matplotlib.rcParams['lines.linewidth'] = 3
cmap = matplotlib.cm.get_cmap('Paired')
colors = cmap(range(8))
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',]
hierarchy.dendrogram(Z, color_threshold=0.2, distance_sort='ascending', labels=labels, link_color_func=lambda k: 'k') #, labels=labels_alphabetical)

ncut = []
for i in range(len(networks)):
    r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
    ncut.append(len(networks[i])-r)
ncut=np.array(ncut)[:L]

plt.yticks(np.arange(np.max(ncut)+.2, 0, -0.5)[::-1], np.max(ncut)+.2-np.arange(np.max(ncut)+.2, 0, -0.5)[::-1])
ax.set_ylabel('Ncut')
ax.set_title('(c) cluster hierarchy')
    
ax = f.add_subplot(gs[0, 0:2])
ax.set_title('(a) particle labels at initial time')
bounds = np.arange(-0.5,L+0.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))

cmap = matplotlib.colors.ListedColormap(colors)   
field_plot = np.ones(G.shape[0]) *(-10000)
for k in range(L): 
    field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label

field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
drifter_data.scatter_position_with_labels_flat(ax, field_plot, cmap=cmap, norm=norm,
                                                  cbar=False, size=1, t=0)

t =plt.annotate('A', (0.6,0.1), xycoords='axes fraction', size=20, color=cmap(0))
t.set_bbox(dict(facecolor='k', alpha=0.7))
    
t =plt.annotate('B', (0.65,0.2), xycoords='axes fraction', size=20, color=cmap(1))
t.set_bbox(dict(facecolor='white', alpha=0.7))

t =plt.annotate('C', (0.3,0.2), xycoords='axes fraction', size=20, color=cmap(2))
t.set_bbox(dict(facecolor='k', alpha=0.5))
    
t =plt.annotate('D', (0.75,0.37), xycoords='axes fraction', size=20, color=cmap(3))
t.set_bbox(dict(facecolor='white', alpha=0.7))

t =plt.annotate('E', (0.15,0.1), xycoords='axes fraction', size=20, color=cmap(4))
t.set_bbox(dict(facecolor='k', alpha=0.5))

t =plt.annotate('F', (0.25,0.4), xycoords='axes fraction', size=20, color=cmap(5))
t.set_bbox(dict(facecolor='white', alpha=0.7))

t =plt.annotate('G', (0.61,0.5), xycoords='axes fraction', size=20, color=cmap(6))
t.set_bbox(dict(facecolor='k', alpha=0.5))

t =plt.annotate('H', (0.75,0.55), xycoords='axes fraction', size=20, color=cmap(7))
t.set_bbox(dict(facecolor='white', alpha=0.7))

ax = f.add_subplot(gs[0, 2:4])
ax.set_title('(b) particle labels at final time')
drifter_data.scatter_position_with_labels_flat(ax, field_plot, cmap=cmap, norm=norm,
                                                  cbar=False, size=1, t=-1)
    
f.savefig('./figures/dg_hierarchical', dpi=300)