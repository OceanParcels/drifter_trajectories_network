# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:29:46 2020

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
import networkx as nx
from create_network import trajectory_data, undirected_network
from datetime import datetime, timedelta


"""
Plots for drifter data set
"""

def plot_number_synchronous_trajectories():
    """
    Plot the number of synchronous drifter trajectories
    """
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID)
       
    drifter_data.start_end_times()
    start_times = drifter_data.start_times
    end_times = drifter_data.end_times
    
    t0_range = np.array([[datetime(y,m,1,0,0) for m in range(1,13)] for y in range(1989, 2020)]).flatten()
    days_range = np.arange(30,750,30)
    days_range = np.array([timedelta(days=int(d)) for d in days_range])
    counts = np.zeros((len(t0_range), len(days_range)))
    
    for k in range(len(time)):
        for i in range(len(t0_range)):    
            t0 = t0_range[i]
            for j in range(len(days_range)):
                d = days_range[j]
                if start_times[k]<=t0 and end_times[k] >=t0 + d:
                    counts[i,j]+=1
    
    x = range(len(t0_range))
    y = np.arange(30,750,30)
    x_ticks = [t0_range[i].strftime("%Y") for i in range(len(t0_range))]
    fig, ax = plt.subplots(figsize = (8,4))
    cmap = ax.pcolormesh(x, y, counts.transpose(), cmap = 'cividis')
    fig.colorbar(cmap)
    plt.xticks(x[::24], x_ticks[::24], rotation='vertical')
    plt.ylabel('Trajectory length (days)', size=11)
    plt.xlabel('Start time', size=11)
    plt.title('Number of synchronous drifter trajectories ', size=12)
    fig.tight_layout()
    plt.savefig('figures/number_of_synchronous_drifter_trajectories', dpi=300)
    


def plot_drifter_data():
    """
    Plot with total drifter measurement counts (map) and histogram
    """
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic_withSST.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                        drifter_time = time, drifter_id = ID)
    drifter_data.compute_symbolic_sequences(d_deg = 1.0, dt_days = 0.25)
    
    sequence = drifter_data.symbolic_sequence
    
    total_symbols = []
    for s in sequence:
        total_symbols += list(s[:])
    
    total_symbols = np.array(total_symbols).astype(int)
    total_symbols = total_symbols[total_symbols>=0]
    unique_symbols, counts = np.unique(total_symbols, return_counts=True)
    d = np.zeros(drifter_data.n_horizontal)
    d[unique_symbols] = counts
    d[d==0] = np.nan
    
    empty_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, drifter_time = time, drifter_id = ID)
    empty_data.set_discretizing_values(d_deg = 1.0)
    
    f = plt.figure(constrained_layout=True, figsize = (8,3.5))
    gs = f.add_gridspec(1, 5)
    
    ax1 = f.add_subplot(gs[0, :3])
    ax1.set_title('Drifter counts', size=12)
    empty_data.plot_discretized_distribution(d, ax1, land=True, cmap='cividis', cbar_orientation ='vertical')
    
    ax2 = f.add_subplot(gs[0, 3:])
    ax2.set_title('Counts per visited cell', size=12)
    h, x = np.histogram(counts, bins=100)
    x_plot = x[0] + np.cumsum(np.diff(x))
    ax2.bar(x_plot[x_plot<5000], h[x_plot<5000], width = np.diff(x[x<5000]), color = 'darkslategrey')
    ax2.grid(True)
    
    plt.savefig('figures/drifter_distribution', dpi=300)



"""
PLots for Tansition matrix method
"""

def plot_spectrum_LPhat():
    
    
    """
    Plot first few eigenvectors and eigenvalues of the Laplacian of P_hat
    """

    P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg1_ddays_60.npz')
    
    #Find strongly connected component and reduce matrix to it. This removes sink cells.
    G = nx.from_scipy_sparse_matrix(P, create_using = nx.DiGraph())
    components = np.array(list(nx.strongly_connected_components(G)))
    component_lengths = np.array([len(s) for s in components])
    component_inds = np.argsort(component_lengths)[::-1]
    components_sorted = components[component_inds]
    component_lengths = np.array([len(c) for c in components_sorted])
    P_reduced = P[list(components_sorted[0]),:][:,list(components_sorted[0])]
    
    #Re-normalize the matrix
    d = np.array(sparse.csr_matrix.sum(P_reduced, axis=1))[:,0]
    D_inv = sparse.diags(1./d)
    P_rn =  (P_reduced.transpose().dot(D_inv)).transpose()
    
    #Find invariant measure
    vals_l, vecs_l = eigs(P_rn.transpose(), k=30, which='LM')
    inds = np.argsort(vals_l)[::-1]
    vals_l = vals_l[inds]
    vecs_l = vecs_l[:, inds]
    pi = np.real(vecs_l[:,0])
    if np.all(pi<=0):
        pi *= -1
    
    #Define symmetric matrix P_hat
    PI = sparse.diags(pi)
    P_W = (PI.dot(P_rn) + P_rn.transpose().dot(PI))/2.
    s = np.array(sparse.csr_matrix.sum(P_W, axis=1))[:,0]
    A = undirected_network(P_W, np.array(list(components_sorted[0])), s)
    A.compute_laplacian_spectrum(K=30)
    
    #Plot a couple of eigenvectors and eigenvalues
    empty_data = trajectory_data()
    empty_data.set_discretizing_values(d_deg=1.0)
    k_s = range(1,6)
    gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1]]
    f = plt.figure(constrained_layout=True, figsize = (10,4.85))
    gs = f.add_gridspec(2, 3)
    
    for i in range(len(k_s)):
        g = gs_s[i]
        k = k_s[i]
        ax1 = f.add_subplot(gs[g[0], g[1]])
    
        field_plot = np.ones(P.shape[0]) *(-10000)
        field_plot[list(components_sorted[0])]= A.eigenvectors[:,k]
        field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
        ax1.set_title(r'n = ' + str(k+1), size=12)
        empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                                  land=True, cmap='cividis', 
                                                  cbar_orientation='vertical')
    
    ax1 = f.add_subplot(gs[1,2])
    ax1.plot(A.eigenvalues, 'ok', )
    ax1.set_title('Smallest eigenvalues', size=12)
    ax1.grid(True)
    ax1.set_ylabel(r'$\lambda_i$', size=12)
    # f.tight_layout()
    plt.savefig('figures/spectrum_Phat', dpi=300)
    


def P_hat_hierarchical_spectral_clustering():
    """
    Hierachical clustering on the Laplacian eigenvectors of P_hat
    """
    
    #Reduce to largest strongly connected component and re-normalize
    P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg1_ddays_60.npz')
    G = nx.from_scipy_sparse_matrix(P, create_using = nx.DiGraph())
    components = np.array(list(nx.strongly_connected_components(G)))
    component_lengths = np.array([len(s) for s in components])
    component_inds = np.argsort(component_lengths)[::-1]
    components_sorted = components[component_inds]
    component_lengths = np.array([len(c) for c in components_sorted])
    P_reduced = P[list(components_sorted[0]),:][:,list(components_sorted[0])]
    d = np.array(sparse.csr_matrix.sum(P_reduced, axis=1))[:,0]    
    D_inv = sparse.diags(1./d)
    P_rn =  (P_reduced.transpose().dot(D_inv)).transpose()
        
    #Find measure pi and construct Phat
    vals, vecs_l = eigs(P_rn.transpose(), k=10, which='LM')
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    vecs_l = vecs_l[:, inds]
    
    pi = np.real(vecs_l[:,0])
    if np.all(pi<=0):
        pi *= -1
    
    PI = sparse.diags(pi)
    P_W = (PI.dot(P_rn) + P_rn.transpose().dot(PI))/2.
    s = np.array(sparse.csr_matrix.sum(P_W, axis=1))[:,0]
    
    networks = {}
    networks[0] = [undirected_network(P_W, np.array(list(components_sorted[0])), s)]
    
    for i in range(1,10):
        
        optimal_drhos = []
        optimal_cutoffs = []
        for nw in networks[i-1]: 
            if nw.N<100: 
                optimal_drhos.append(np.nan)
                optimal_cutoffs.append(np.nan)
                continue
    
            nw.compute_laplacian_spectrum()
            V_fiedler = nw.eigenvectors[:,1]
            c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
            
            drhos = []
            for c in c_range:
            
                indices_1 = np.argwhere(V_fiedler<=c)[:,0]
                indices_2 = np.argwhere(V_fiedler>c)[:,0]
                drhos.append(nw.drho_split(indices_1, indices_2))
        
            drhos = np.array(drhos)
            plt.plot(c_range, drhos)
            plt.yscale('log')
            plt.grid(True)
            plt.title(r'$\Delta \rho_{global}$ for different cutoffs')
            plt.show()
            cutoff_opt = c_range[np.nanargmax(drhos)]
            print('Choosing as cutoff: ', str(cutoff_opt))
            
            optimal_drhos.append(np.nanmax(drhos))
            optimal_cutoffs.append(cutoff_opt)
        
        i_cluster = np.nanargmax(optimal_drhos)
        print('Splitting cluster ', i_cluster+1)
        cutoff_cluster = optimal_cutoffs[np.nanargmax(optimal_drhos)]
        nw_to_split = networks[i-1][i_cluster]
        V_fiedler = nw_to_split.eigenvectors[:,1]
        c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
        indices_1 = np.argwhere(V_fiedler<=cutoff_cluster)[:,0]
        indices_2 = np.argwhere(V_fiedler>cutoff_cluster)[:,0]
        
        adjacency_matrix_1 = nw_to_split.adjacency_matrix[indices_1, :][:, indices_1]
        adjacency_matrix_2 = nw_to_split.adjacency_matrix[indices_2, :][:, indices_2]
        cluster_indices_1 = nw_to_split.cluster_indices[indices_1]
        cluster_indices_2 = nw_to_split.cluster_indices[indices_2]
        cluster_volume_1 = nw_to_split.cluster_volume[indices_1]
        cluster_volume_2 = nw_to_split.cluster_volume[indices_2]
        cluster_label_1 = nw_to_split.cluster_label + '0'
        cluster_label_2 = nw_to_split.cluster_label + '1'
        
        network_children = [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                        undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
        
        networks[i] = networks[i-1].copy()
        networks[i].pop(i_cluster)
        networks[i] += network_children
        
        field_plot = np.zeros(P.shape[0])
        for k in range(len(networks[i])):
            field_plot[networks[i][k].cluster_indices]= k+1
        
        empty_data = trajectory_data()
        empty_data.set_discretizing_values(d_deg=1.0)
        
        f = plt.figure(constrained_layout=True, figsize = (15,7))
        gs = f.add_gridspec(1, 1)
        ax1 = f.add_subplot(gs[0, 0])
        ax1.set_title('Clusters step ' + str(i), size=16)
        empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                                  land=True, cmap='cividis', 
                                                  cbar_orientation='vertical')
        plt.show()

    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values(d_deg=1.0)
    gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2],
            [2,0], [2,1]]
    f = plt.figure(constrained_layout=True, figsize = (10,8.5))
    gs = f.add_gridspec(3, 3)
    
    for i in range(1,len(gs_s)+1):
        g = gs_s[i-1]
        ax1 = f.add_subplot(gs[g[0], g[1]])
    
        field_plot = np.ones(P.shape[0]) *(-10000)
        
        for k in range(len(networks[i])):
            field_plot[networks[i][k].cluster_indices]= k
            
        # field_plot[list(components_sorted[0])]= A.eigenvectors[:,k]
        field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
        ax1.set_title(r'K = ' + str(k+1), size=12)
        empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                                  land=True, cmap='cividis', colbar=False)
    
    
    
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
        
    ax1 = f.add_subplot(gs[2,2])
    ax1.plot(ncut, 'ok', )
    ax1.set_title('Ncut', size=12)
    ax1.grid(True)
    
    plt.savefig('figures/Phat_cLustering', dpi=300)


def P_out_degree():
    """
    Hierachical clustering on the Laplacian eigenvectors of P_hat
    """
    
    #Reduce to largest strongly connected component and re-normalize
    P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg1_ddays_60.npz')
    P.data[:]=1
    
    out_degree = np.array(sparse.csr_matrix.sum(P, axis=1))[:,0]
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values(d_deg=1.0)
   
    
    vals, vecs_l = eigs(P.transpose(), k=10, which='LM')
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    vecs_l = vecs_l[:, inds]
    
    pi = np.real(vecs_l[:,0])
    if np.all(pi<=0):
        pi *= -1
        
    f = plt.figure(constrained_layout=True, figsize = (10,5))
    gs = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(gs[0,0])
    ax1.set_title('Invariant measure of P',size=12)
    empty_data.plot_discretized_distribution(pi, ax1, logarithmic=False,
                                              land=True, cmap='cividis', colbar=True,
                                              cbar_orientation = 'vertical')
    ax1 = f.add_subplot(gs[0,1])    
    ax1.set_title('Out degree of P',size=12)
    empty_data.plot_discretized_distribution(out_degree, ax1, logarithmic=False,
                                              land=True, cmap='cividis', colbar=True,
                                              cbar_orientation = 'vertical')

    
    plt.savefig('figures/out_degree_stationary_density', dpi=300)
