# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:21:01 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from create_network import trajectory_data, undirected_network, bipartite_network, construct_dendogram
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.colors

d_deg=[1.0]
max_length = 365
particle_network = True
bin_network = False
adjacency_matrix_type = 'random_walk_matrix_'

domain_edges = (-100., 80, 0, 85) # full domain, to cover all points along any trajectory    
drifter_data = trajectory_data.from_npz('drifter_data_north_atlantic/drifterdata_north_atlantic.npz',
                                         time_interval_days=0.25, n_step = 4, domain_type = "north_atlantic_domain",
                                         domain_edges = domain_edges, set_nans=False)

if max_length is not None:
    drifter_data.set_max_length(max_length = max_length)

# drifter_data.set_min_length(min_length = 20)

drifter_data.set_nans(constrain_to_domain=True) #To enforce entire data on north atlantic
drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = 1, 
                                            trajectory_segments=False)

trajectory_lenghts = [len(s) for s in drifter_data.symbolic_sequence]
T = np.max(trajectory_lenghts)

#Compute C(t) matrices
C = []
for i in range(T): C.append(drifter_data.compute_C(i))

path_network = C[0].copy()
for i in range(1, T): path_network += C[i]

if particle_network:
    #particle space
    B = bipartite_network(path_network)
    # if adjacency_matrix_type == 'stochastic_complement':
    #     A = undirected_network(B.stochastic_complement_adjacency_matrix(space = 'X'))
    # else:
    #     A = undirected_network(B.projection_adjacency_matrix(space = 'X'))
    
    R = B.random_walk_matrix()
    A = undirected_network(R.dot(R.transpose()))
    A.connected_components()
    A1 = A.sub_networks[0]
    w, v = A1.compute_laplacian_spectrum(K=40)
    
    K = 20
    A1.hierarchical_clustering_ShiMalik(K)    
    networks = A1.clustered_networks
    
    L=20
    Z = construct_dendogram([networks[i] for i in range(L)])
        
    f = plt.figure(constrained_layout=True, figsize = (10,4))
    gs = f.add_gridspec(1, 2)
    ax = f.add_subplot(gs[0, 1])
    
    matplotlib.rcParams['lines.linewidth'] = 3
    
    colors2 = np.array(['midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'darkred', 'darkslategrey', 'sienna', 'limegreen', 'orange', 'darkorchid', 
                        'yellow', 'black', 'aqua', 'crimson', 'indigo', 'dimgray', 'darkviolet',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown'])
    
    hierarchy.dendrogram(Z, color_threshold=0.2, distance_sort='ascending', link_color_func=lambda k: 'k') #, labels=labels_alphabetical)
    
    ax.set_xlabel('Cluster')
    
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
    
    ncut=np.array(ncut)[:L]
    
    plt.yticks(np.arange(np.max(ncut)+.2, 0, -0.5)[::-1], np.max(ncut)+.2-np.arange(np.max(ncut)+.2, 0, -0.5)[::-1])
    ax.set_ylabel('Ncut')
    ax = f.add_subplot(gs[0, 0])
    colors = np.array(['midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'darkred', 'darkslategrey', 'sienna', 'limegreen', 'orange', 'darkorchid', 
                        'yellow', 'black', 'aqua', 'crimson', 'indigo', 'dimgray', 'darkviolet',])
    bounds = np.arange(-0.5,L+0.5,1)
    # bounds = list(range(K+1))
    
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    cmap = matplotlib.colors.ListedColormap(colors)    
    
    field_plot = np.ones(path_network.shape[0]) *(-10000)
        
    for k in range(L):
        field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label
        
    # field_plot[list(components_sorted[0])]= A.eigenvectors[:,k]
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
    drifter_data.scatter_initial_position_with_labels(field_plot, ax, cmap=cmap, norm=norm,
                                                      cbarticks = list(range(K)), size=10)
    
    left, bottom, width, height = [0.32, 0.2, 0.15, 0.4]
    ax2 = f.add_axes([left, bottom, width, height])
    
    # ax2s = f.add_subplot(gs[2, 1])
    ax2.plot(range(L), w[:L], 'o', color='maroon', markersize=2)
    ax2.plot(range(L,30), w[L:30], 'o', color='k', markersize=2)
    ax2.tick_params(axis="y",direction="in", pad=-27)
    xticks = [round(n,2) for n in np.arange(0.05,0.3,0.1)]
    plt.yticks(xticks, xticks)
    # ax2.set_facecolor('xkcd:mint green')
    plt.xticks([])
    plt.grid(True)
    ax.set_ylabel(r'$\lambda$')
    
    f.savefig('./figures/north_atlantic/north_atlantic_' + adjacency_matrix_type + 'particles_maxlen_' + str(max_length), dpi=300)
    


if bin_network:
    #bin space
    B = bipartite_network(path_network)
    if adjacency_matrix_type == 'stochastic_complement':
        A = undirected_network(B.stochastic_complement_adjacency_matrix(space = 'Y'))
    else:
        A = undirected_network(B.projection_adjacency_matrix(space = 'Y'))
    
    A.connected_components()
    A1 = A.sub_networks[0]
    w, v = A1.compute_laplacian_spectrum(K=40)
    
    K = 25
    A1.hierarchical_clustering_ShiMalik(K)    
    networks = A1.clustered_networks
    
    L=13
    Z = construct_dendogram([networks[i] for i in range(L)])
        
    f = plt.figure(constrained_layout=True, figsize = (10,4))
    gs = f.add_gridspec(1, 2)
    ax = f.add_subplot(gs[0, 1])
    
    matplotlib.rcParams['lines.linewidth'] = 3
    
    colors2 = np.array(['midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'darkred', 'darkslategrey', 'sienna', 'limegreen', 'orange', 'darkorchid', 
                        'yellow', 'black', 'aqua', 'crimson', 'indigo', 'dimgray', 'darkviolet',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown'])
    
    hierarchy.dendrogram(Z, color_threshold=0.2, distance_sort='ascending', link_color_func=lambda k: 'k') #, labels=labels_alphabetical)
    
    ax.set_xlabel('Cluster')
    
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
    
    ncut=np.array(ncut)[:L]
    
    plt.yticks(np.arange(np.max(ncut)+.2, 0, -0.5)[::-1], np.max(ncut)+.2-np.arange(np.max(ncut)+.2, 0, -0.5)[::-1])
    ax.set_ylabel('Ncut')
    ax = f.add_subplot(gs[0, 0])
    colors = np.array(['midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown',
                        'darkred', 'darkslategrey', 'sienna', 'limegreen', 'orange', 'darkorchid', 
                        'yellow', 'black', 'aqua', 'crimson', 'indigo', 'dimgray', 'darkviolet',])
    bounds = np.arange(-0.5,L+0.5,1)
    # bounds = list(range(K+1))
    
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    cmap = matplotlib.colors.ListedColormap(colors)    
    
    field_plot = np.ones(path_network.shape[1]) *(-10000)
        
    for k in range(L):
        field_plot[networks[L-1][k].cluster_indices]= networks[L-1][k].cluster_label
        
    # field_plot[list(components_sorted[0])]= A.eigenvectors[:,k]
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
    drifter_data.plot_discretized_distribution(field_plot, ax, cmap=cmap, norm=norm) 
                                                      # cbarticks = list(range(K)), size=10)
    
    left, bottom, width, height = [0.32, 0.2, 0.15, 0.4]
    ax2 = f.add_axes([left, bottom, width, height])
    
    # ax2s = f.add_subplot(gs[2, 1])
    ax2.plot(range(L), w[:L], 'o', color='maroon', markersize=2)
    ax2.plot(range(L,30), w[L:30], 'o', color='k', markersize=2)
    ax2.tick_params(axis="y",direction="in", pad=-27)
    xticks = [round(n,2) for n in np.arange(0.05,0.3,0.1)]
    plt.yticks(xticks, xticks)
    # ax2.set_facecolor('xkcd:mint green')
    plt.xticks([])
    plt.grid(True)
    ax.set_ylabel(r'$\lambda$')
    
    f.savefig('./figures/north_atlantic/north_atlantic_' + adjacency_matrix_type + 'bins_maxlen_' + str(max_length), dpi=300)