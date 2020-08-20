"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Script for clustering of Noarth Atlantic drifter data set in particle and bin space
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data, undirected_network, bipartite_network, construct_dendrogram
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.colors

"""
Here are four blocks for different values of bin size and trajectory length. Uncomment the block you need. 
In the end of this script, we generate fig. C2 from the saved clustering labels.
"""

# #dx = dy = 1 degree, limit to 365 days
# plot_full=True
# colors = np.array(['midnightblue', 'dimgray', 'dimgray', 'dimgray', 'sienna', 'skyblue', 'dimgray',
#                        'firebrick', 'dimgray', 'dodgerblue', 'aqua', 'dimgray', 'dimgray', 
#                        'mediumaquamarine', 'teal', 'dimgray', 'coral','dimgray','orange','purple'])
# K = 20 #number of eigenvectors to be computed
# L=20 #Level of hierarchical clustering
# d_deg=[1.0]
# max_length = 365


# #dx = dy = 1 degree, no limit on trajectory length
# plot_full=False
# #colors are adjusted so that they match the plot_365days=True case
# colors = np.array(['midnightblue', 'dimgray', 'dimgray', 'sienna', 'dimgray', 'skyblue', 'dimgray',
#                     'dimgray', 'dimgray', 'aqua', 'firebrick', 'dodgerblue', 'coral', 
#                     'dimgray', 'dimgray', 'dimgray', 'teal','dimgray','dimgray','dimgray'])
# K = 20 #number of eigenvectors to be computed
# L=16 #Number of clusters in hierarchical clustering. This is chosen such that the max NCut is similar to option 1 (around 3.6)
# d_deg=[1.0]
# max_length = None
# figname = './figures/na_clusters_tmaxinf'


# dx = dy = 2 degree
# plot_full=False
# colors = np.array(['teal', 'dimgray', 'dimgray', 'midnightblue', 'sienna', 'dimgray', 'dimgray',
#                     'skyblue', 'firebrick', 'dimgray', 'midnightblue', 'aqua', 'dimgray', 
#                     'dodgerblue', 'dimgray', 'mediumaquamarine', 'orange','k','coral','dimgray'])
# K = 20 #number of eigenvectors to be computed
# L=20 #Number of clusters in hierarchical clustering. This is chosen such that the max NCut is similar to option 1 (around 3.6)
# d_deg=[2.0]
# max_length = 365
# figname = './figures/na_clusters_tma365_d_deg2'


# dx = dy = 4 degree
plot_full=False
colors = np.array(['teal', 'dimgray', 'dimgray', 'sienna', 'sienna', 'dimgray', 'skyblue',
                    'firebrick', 'midnightblue', 'aqua', 'dodgerblue', 'orange', 'dimgray', 
                    'sienna', 'dimgray', 'k', 'coral','purple','mediumaquamarine','dimgray'])
K = 20 #number of eigenvectors to be computed
L=20 #Number of clusters in hierarchical clustering. This is chosen such that the max NCut is similar to option 1 (around 3.6)
d_deg=[4.0]
max_length = 365
figname = './figures/na_clusters_tma365_d_deg4'

#Load daily data
drifter_data = trajectory_data.from_npz('drifterdata_north_atlantic.npz',
                                          time_interval=0.25, n_step = 4, domain_type = "north_atlantic_domain")

if max_length is not None: drifter_data.set_max_length(max_length = max_length)

drifter_data.compute_symbolic_itineraries(bin_size = d_deg, dt = 1)
trajectory_lenghts = [len(s) for s in drifter_data.symbolic_sequence]
T = np.max(trajectory_lenghts)

#Compute C and G matrices
C = []
for i in range(T): C.append(drifter_data.compute_C(i))
G = C[0].copy()
for i in range(1, T): G += C[i]

#Set up networks and hierarchical clustering according to Shi&Malik 2000
B = bipartite_network(G)
A = undirected_network(B.projection_adjacency_matrix(space = 'X'))    
A1 = A.largest_connected_component()
w, v = A1.compute_laplacian_spectrum(K=40)
A1.hierarchical_clustering_ShiMalik(K, plots=False)    
networks = A1.clustered_networks

if plot_full:
    Z = construct_dendrogram([networks[i] for i in range(L)])
    
    f = plt.figure(constrained_layout=True, figsize = (10,8))
    gs = f.add_gridspec(2, 4)
    ax = f.add_subplot(gs[1, 1:3])
    matplotlib.rcParams['lines.linewidth'] = 3
    labels = ['A', '.','.','.','B','C','.','D','.','E','F','.','.','G','H','.','I','.','J','K']
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
    for k in range(L): field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    drifter_data.scatter_position_with_labels_geo(ax, field_plot, cmap=cmap, norm=norm,
                                                      cbar=False, size=10, t=0)

    t = plt.annotate('A', (0.25,0.2), xycoords='axes fraction', size=18, color='midnightblue')
    t.set_bbox(dict(facecolor='w', alpha=0.7))
    
    t =plt.annotate('B', (0.36,0.56), xycoords='axes fraction', size=18, color='sienna')
    t.set_bbox(dict(facecolor='w', alpha=0.7))
    
    t =plt.annotate('C', (0.65,0.05), xycoords='axes fraction', size=18, color='skyblue')
    t.set_bbox(dict(facecolor='gray', alpha=0.8))
    
    t =plt.annotate('D', (0.7,0.65), xycoords='axes fraction', size=18, color='firebrick')
    t.set_bbox(dict(facecolor='w', alpha=0.7))
    
    t =plt.annotate('E', (0.32,0.34), xycoords='axes fraction', size=18, color='dodgerblue')
    t.set_bbox(dict(facecolor='w', alpha=0.7))
        
    t =plt.annotate('F', (0.02,0.28), xycoords='axes fraction', size=18, color='aqua')
    t.set_bbox(dict(facecolor='w', alpha=0.3))
    
    t =plt.annotate('G', (0.52,0.14), xycoords='axes fraction', size=18, color='mediumaquamarine')
    t.set_bbox(dict(facecolor='gray', alpha=0.8))
    
    t =plt.annotate('H', (0.5,0.3), xycoords='axes fraction', size=18, color='teal')
    t.set_bbox(dict(facecolor='w', alpha=0.7))    
    
    t =plt.annotate('I', (0.48,0.68), xycoords='axes fraction', size=18, color='coral')
    t.set_bbox(dict(facecolor='w', alpha=0.7))
    
    t =plt.annotate('J', (0.92,0.8), xycoords='axes fraction', size=18, color='orange')
    t.set_bbox(dict(facecolor='gray', alpha=0.8))
    
    t =plt.annotate('K', (0.7,0.33), xycoords='axes fraction', size=18, color='purple')
    t.set_bbox(dict(facecolor='w', alpha=0.7))

    ax = f.add_subplot(gs[0, 2:4])
    ax.set_title('(b) particle labels at final time')
    drifter_data.scatter_position_with_labels_geo(ax, field_plot, cmap=cmap, norm=norm,
                                                      cbar=False, size=10, t=-1)
    
    # f.savefig('./figures/na_clusters_tmax365', dpi=300)
    
    #Plot spectrum with lines corresponding to known regions
    f, ax = plt.subplots(figsize = (7,5)) 
    ax.plot(range(30), w[:30], 'o', color='k', markersize=4)
    
    ax.grid(True)
    ax.set_ylabel(r'$\lambda$')
    ax.set_title(r'Spectrum of $L_s$', size=14) 
    
    # f.savefig('./figures/na_clusters_tmax365_spectrum', dpi=300, bbox_inches='tight')
    
else:
    f = plt.figure(constrained_layout=True, figsize = (5,4))
    gs = f.add_gridspec(1, 1)
    
    ax = f.add_subplot(gs[0, 0])
    ax.set_title('Particle labels at initial time')
    
    bounds = np.arange(-0.5,L+0.5,1)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    cmap = matplotlib.colors.ListedColormap(colors)    
    field_plot = np.ones(G.shape[0]) *(-10000)
    for k in range(L): field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label
    
    np.save(figname, field_plot)
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    drifter_data.scatter_position_with_labels_geo(ax, field_plot, cmap=cmap, norm=norm,
                                                      cbar=False, size=10, t=0)
    f.savefig(figname, dpi=300)



# # plot clustering of dx=2 and dx=4 together
# max_length = 365

# drifter_data = trajectory_data.from_npz('drifterdata_north_atlantic.npz',
#                                           time_interval=0.25, n_step = 4, domain_type = "north_atlantic_domain")

# if max_length is not None: drifter_data.set_max_length(max_length = max_length)

# colors_deg2 = np.array(['teal', 'dimgray', 'dimgray', 'midnightblue', 'sienna', 'dimgray', 'dimgray',
#                     'skyblue', 'firebrick', 'dimgray', 'midnightblue', 'aqua', 'dimgray', 
#                     'dodgerblue', 'dimgray', 'mediumaquamarine', 'orange','k','coral','dimgray'])

# colors_deg4 = np.array(['teal', 'dimgray', 'dimgray', 'sienna', 'dimgray', 'dimgray', 'skyblue',
#                     'firebrick', 'midnightblue', 'aqua', 'dodgerblue', 'orange', 'dimgray', 
#                     'mediumaquamarine', 'dimgray', 'k', 'coral','purple','darkgreen','dimgray'])

# labels_deg2 = np.load('./figures/na_clusters_tma365_d_deg2.npy')
# labels_deg4 = np.load('./figures/na_clusters_tma365_d_deg4.npy')

# f = plt.figure(constrained_layout=True, figsize = (10,4))
# gs = f.add_gridspec(1, 2)

# ax = f.add_subplot(gs[0, 0])
# ax.set_title(r'a) Clustering result for $\Delta x = \Delta y = 2^\circ$')
# L=20
# bounds = np.arange(-0.5,L+0.5,1)
# norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
# cmap = matplotlib.colors.ListedColormap(colors_deg2)
# labels_deg2 = np.ma.masked_array(labels_deg2, labels_deg2==-10000)   
# drifter_data.scatter_position_with_labels_geo(ax, labels_deg2, cmap=cmap, norm=norm,
#                                                       cbar=False, size=10, t=0)

# ax = f.add_subplot(gs[0, 1])
# ax.set_title(r'b) Clustering result for $\Delta x = \Delta y = 4^\circ$')
# L=20
# bounds = np.arange(-0.5,L+0.5,1)
# norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
# cmap = matplotlib.colors.ListedColormap(colors_deg4)
# labels_deg4 = np.ma.masked_array(labels_deg4, labels_deg4==-10000)   
# drifter_data.scatter_position_with_labels_geo(ax, labels_deg4, cmap=cmap, norm=norm,
#                                                       cbar=False, size=10, t=0)
# f.savefig('./figures/na_clusteringdx2_dx4', dpi=300)
