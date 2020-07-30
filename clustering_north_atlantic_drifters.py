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
2 options: 
    1. tmax = 365 days, plot for initial time, final time and dendogram. Spectrum. (fugures 6 and 7)
    2. tmax = inf initial time.  (fig. C1)
"""
plot_365days=True #True for option 1

if plot_365days:
    colors = np.array(['midnightblue', 'dimgray', 'dimgray', 'dimgray', 'sienna', 'skyblue', 'dimgray',
                        'firebrick', 'dimgray', 'dodgerblue', 'aqua', 'dimgray', 'dimgray', 
                        'mediumaquamarine', 'teal', 'dimgray', 'coral','dimgray','orange','purple'])
    K = 20 #number of eigenvectors to be computed
    L=20 #Level of hierarchical clustering
    d_deg=[1.0]
    max_length = 365
    plot_paper = True
    plot_labels = True
    plot_spectrum = True

else:
    #colors are adjusted so that they match the plot_365days=True case
    colors = np.array(['midnightblue', 'dimgray', 'dimgray', 'sienna', 'dimgray', 'skyblue', 'dimgray',
                        'dimgray', 'dimgray', 'aqua', 'firebrick', 'dodgerblue', 'coral', 
                        'dimgray', 'dimgray', 'dimgray', 'teal','dimgray','dimgray','dimgray'])
    K = 20 #number of eigenvectors to be computed
    L=16 #Number of clusters in hierarchical clustering. This is chosen such that the max NCut is similar to option 1 (around 3.6)
    d_deg=[1.0]
    max_length = None
    t_plot = 0
    plot_dendrogram = False
    plot_spectrum = False
    plot_labels=False
    
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
A1.hierarchical_clustering_ShiMalik(K)    
networks = A1.clustered_networks

if plot_365days:
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
    
    f.savefig('./figures/na_clusters_tmax365', dpi=300)
    
    #Plot spectrum with lines corresponding to known regions
    # labels = np.array(labels)
    # inds_deleted = np.argwhere(labels=='.')[:,0]
    f, ax = plt.subplots(figsize = (7,5)) 
    ax.plot(range(30), w[:30], 'o', color='k', markersize=4)
    
    # x_used_significant = np.array([i for i in range(L) if i not in inds_deleted])
    # ax.plot(x_used_significant, w[x_used_significant], 'o', color='maroon', markersize=4)
    # ax.plot(inds_deleted, w[inds_deleted], 'o', color='darkgrey', markersize=4)
    # ax.plot(range(L,30), w[L:30], 'o', color='k', markersize=4)
    ax.grid(True)
    # ax.axvline(4.5, color = 'midnightblue', linestyle = '--', linewidth = 1.5, label='Subtropical | Subpolar Gyre')
    # ax.axvline(7.5, color = 'sienna', linestyle = '--', linewidth = 1.5, label='Subpolar Gyre | Nordic Seas')
    # ax.axvline(9.5, color = 'skyblue', linestyle = '--', linewidth = 1.5, label = 'Northern | Southern Subtropics')
    # ax.axvline(10.5, color = 'aqua', linestyle = '--', linewidth = 1.5, label = 'Caribbean Sea')
    # ax.axvline(14.5, color = 'dodgerblue', linestyle = '--', linewidth = 1.5, label = 'Western Boundary Current incl. Bay of Biscay')
    # ax.axvline(16.5, color = 'coral', linestyle = '--', linewidth = 1.5, label = 'Greenland Current')
    # ax.axvline(18.5, color = 'orange', linestyle = '--', linewidth = 1.5, label = 'Barents Sea')
    # ax.axvline(19.5, color = 'purple', linestyle = '--', linewidth = 1.5, label = 'Bay of Biscay')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=False, ncol=2)
    ax.set_ylabel(r'$\lambda$')
    ax.set_title(r'Spectrum of $L_s$', size=14) 
    
    f.savefig('./figures/na_clusters_tmax365_spectrum', dpi=300, bbox_inches='tight')
    
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
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    drifter_data.scatter_position_with_labels_geo(ax, field_plot, cmap=cmap, norm=norm,
                                                      cbar=False, size=10, t=0)
    f.savefig('./figures/na_clusters_tmaxinf', dpi=300)