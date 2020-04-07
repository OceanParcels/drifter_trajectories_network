# -*- coding: utf-8 -*-
"""
Script to do all analysis and create all figures. Make sure that these directories exist:
- Plots are saved in local directory './figures'
- other output saved in './analysis_output'
- north atlantic data set in './drifter_data_north_atlantic'
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



def plot_number_synchronous_trajectories_monthly():
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
    
    x = range(12)
    y = np.arange(30,750,30)
    x_ticks = [t0_range[i].strftime("%Y") for i in range(len(t0_range))]
    
    c_month = np.zeros((12,24))
    for i in range(12):
        c_month[i] = np.sum(counts[i::12,:], axis=0)
    
    
    
    f = plt.figure(constrained_layout=True, figsize = (8,10))
    gs = f.add_gridspec(2, 1)
    
    ax1 = f.add_subplot(gs[0, 0])
    cmap = ax1.pcolormesh(x, y, c_month.transpose(), cmap = 'cividis')
    plt.colorbar(cmap)
    plt.xticks(np.arange(0,11,1), np.arange(1,12,1))
    plt.ylabel('Trajectory length (days)', size=11)
    plt.xlabel('Start time', size=11)
    plt.title('Number of synchronous drifter trajectories per month', size=12)
    
    ax2 = f.add_subplot(gs[1, 0])
    m_range = [6,12]
    for m in m_range:
        ax2.plot(np.arange(0,12,1), c_month[:,m-1], label=str(m) + ' months')
    plt.xticks(np.arange(0,11,1), np.arange(1,12,1))
    plt.ylabel('Trajectory length (days)', size=11)
    plt.xlabel('Start time', size=11)
    plt.title('Number of synchronous drifter trajectories per month', size=12)
    plt.legend()
    # fig.tight_layout()
    #     # plt.savefig('figures/number_of_synchronous_drifter_trajectories', dpi=300)
    
    


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
    
    # plt.savefig('figures/drifter_distribution', dpi=300)


"""
Tansition matrix method (P)
"""

def compute_drifter_transitionmatrix(d_deg=1.0, dt_days=60):
    """
    Computes the transition matrix
    """
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
        
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID)
    drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = dt_days)
    
    print("computing transition matrix")
    drifter_data.compute_P(matrix_type = 'Markov')
    sparse.save_npz('analysis_output/transition_matrix_markov_ddeg_' + str(d_deg) + '_dt_' 
            + str(dt_days), drifter_data.P)


def plot_spectrum_LPhat():
    """
    Plot first few eigenvectors and eigenvalues of the Laplacian of P_hat
    """
    
    # P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg1_ddays_60.npz')
    P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg_0.5_dt_10.npz')
    
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
    empty_data.set_discretizing_values(d_deg=0.5)
    k_s = range(1,6)
    gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1]]
    f = plt.figure(constrained_layout=True, figsize = (10,4.85))
    gs = f.add_gridspec(2, 3)
    
    for i in range(len(k_s)):
        g = gs_s[i]
        k = k_s[i]
        ax1 = f.add_subplot(gs[g[0], g[1]])
    
        field_plot = np.ones(P.shape[0]) *(-10000)
        field_plot[list(components_sorted[0])]= A.Lsym_eigenvectors[:,k]
        field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
        ax1.set_title(r'n = ' + str(k+1), size=12)
        empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                                  land=True, cmap='cividis', 
                                                  cbar_orientation='vertical')
    
    ax1 = f.add_subplot(gs[1,2])
    ax1.plot(A.Lsym_eigenvalues, 'ok', )
    ax1.set_title('Smallest eigenvalues', size=12)
    ax1.grid(True)
    ax1.set_ylabel(r'$\lambda_i$', size=12)
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
    
    A = undirected_network(P_W, np.array(list(components_sorted[0])), s)
    
    A.hierarchical_clustering_ShiMalik(10)
    
    networks = A.clustered_networks
    
    colors = np.array(['midnightblue', 'skyblue', 'firebrick', 'b', 'salmon', 'olivedrab', 'rosybrown','darkred', 'darkslategrey']) #, 'sienna'])
    
    
    bounds = list(range(10))
    import matplotlib.colors
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    cmap = matplotlib.colors.ListedColormap(colors)    
    
        
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
            field_plot[networks[i][k].cluster_indices]= networks[i][k].cluster_label
            
        # field_plot[list(components_sorted[0])]= A.eigenvectors[:,k]
        field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
        ax1.set_title(r'K = ' + str(k+1), size=12)
        empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                                  land=False, colbar=False,
                                                  cmap = cmap, norm=norm)
    
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
        
    ax1 = f.add_subplot(gs[2,2])
    ax1.plot(ncut, 'ok', )
    ax1.set_title('Ncut', size=12)
    ax1.grid(True)
    
    plt.savefig('figures/Phat_cLustering', dpi=300)



def P_hat_hierarchical_spectral_clustering_individual_rhos():
    """
    Hierachical clustering, with cutoff on indidual cohernece ratios (this has not yielded big differences to 
    hierarchical clustering optimizing the global coherence ratio)
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
    
    A = undirected_network(P_W, np.array(list(components_sorted[0])), s)
    
    A.hierarchical_clustering_individual_rhos(0.9)
    
    networks = A.networks
    
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

    

"""
Path intersections
"""
def compute_path_intersection_matrix(d_deg, n_t):
    
    d_deg = 1
    n_t = 100
    print('Computing path intersections for d_deg = ', d_deg)
    print('n_t = ', n_t)
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID)
    sub_drifter_data = drifter_data.compute_subpaths_data(path_days=100)
    
    sub_drifter_data.compute_symbolic_sequences(d_deg = d_deg, dt_days = 0.25)
    
    sub_drifter_data.compute_C(n_t=n_t)
    
    
    # unique_sequences = drifter_data.unique_sequences
    
    np.savez('analysis_output/path_intersection_matrix_counts_d_deg' + str(d_deg) + '_nt_' + str(n_t), 
              C = sub_drifter_data.C.toarray(), indices = sub_drifter_data.drifter_indices_C)
    
    # for n_t in [10, 30, 50, 100, 150, 200, 300, 500, 700]:
# compute_path_intersection_matrix(d_deg = 0.5, n_t=300)




# def plot_n_intersections():
    
# C_counts = sparse.load_npz('analysis_output/path_intersection_matrix_counts.npz').toarray()
# D = np.diag(np.diagonal(C_counts))
# C_others = C_counts - D

# intersections = C_others[C_others!=0].flatten()

# plt.hist(intersections[intersections<150], bins=50)


    # C_counts 


def cluster_path_intersection_network(d_deg, n_intersections=1):
    
    d_deg = 1
    n_t = 100
    
    data = np.load('analysis_output/path_intersection_matrix_counts_d_deg' + str(d_deg) 
                   + '_nt_' + str(n_t) + '.npz')
    C = sparse.csr_matrix(data['C'])
    indices_C = data['indices']
    
    C_counts = C.toarray()
    D = np.diag(np.diagonal(C_counts))
    C_others = C_counts - D
    
    intersections = C_others[C_others!=0].flatten()
    
    x, y = np.unique(intersections, return_counts=True)
    plt.plot(x,y)
    
    # np.histogram(intersections, bins=20)
    plt.show()
    
    # C_counts = sparse.load_npz('analysis_output/path_intersection_matrix_counts.npz')
    B = C + C.transpose() - sparse.diags(C.diagonal())
    
    B.data[:] = 1
    # B.data[:][B.data[:]<5]=0
    # B.data[:][B.data[:]>=5]=1
    # r = np.array(C.diagonal())
    # # c =  np.array(sparse.csr_matrix.sum(B, axis=0))[0,:]
    
    # R_sqrt_inv = sparse.diags(1./np.sqrt(r)).tocsr()
    # # C_sqrt_inv = sparse.diags(1./np.sqrt(c))
    
    # C_norm = R_sqrt_inv.dot(B).dot(R_sqrt_inv)
    
    
    A = undirected_network(B, cluster_indices = indices_C)
    A.connected_components()
    
    A_component1 = A.sub_networks[0]
    
    A_component1.compute_laplacian_spectrum()
    plt.plot(A_component1.Lsym_eigenvalues, 'o')
    
    A_component1.cluster_kmeans(K=3)
    labels = A_component1.kmeans_labels
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
       
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                            drifter_time = time, drifter_id = ID)
    
    drifter_data.compute_symbolic_sequences(d_deg=d_deg, dt_days = 0.25)
    
    gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2], 
            [3,0], [3,1], [3,2], [4,0], [4,1], [4,2]]
    f = plt.figure(constrained_layout=True, figsize = (10,8.5))
    gs = f.add_gridspec(5, 3)
    
    for i in range(0,len(gs_s)):
        g = gs_s[i]
        ax1 = f.add_subplot(gs[g[0], g[1]])
        indices = A_component1.cluster_indices[np.argwhere(labels == i)[:,0]]
        # networks_sorted[i].cluster_indices
        ax1.set_title('Cluster ' + str(i+1) + ' (' + str(len(indices)) + ' trajectories)', size=12)
        sub_drifter_data.trajectories_density(indices=indices, nmax=n_t)
        sub_drifter_data.plot_discretized_distribution(sub_drifter_data.d_cluster, ax1, logarithmic=True)
        
            
        ax1 = f.add_subplot(gs[2,0])
        ax1.plot(range(1, K+1), A_component1.Lsym_eigenvalues[:K], 'o', color = 'maroon')
        ax1.plot(range(K+1, len(A_component1.Lsym_eigenvalues)+1), A_component1.Lsym_eigenvalues[K:], 'ok', )
        ax1.set_title('Laplacian eigenvalues', size=12)
        ax1.grid(True)  
        

    
    
    n_intersections=5
    
    
    C.data[:][C .data[:]<n_intersections]=0
    C.data[:][C .data[:]>=n_intersections]=1
    
    A = undirected_network(C + C .transpose())
    A.connected_components()
    
    A_component1 = A.sub_networks[0]
    
    A_component1.compute_laplacian_spectrum()
    plt.plot(A_component1.Lsym_eigenvalues, 'o')
        

    
    K=5
    
    A_component1.hierarchical_clustering_ShiMalik(K)
    networks = np.array(A_component1.clustered_networks[K-1])
    labels = [nw.cluster_label for nw in networks]
    networks_sorted = networks[np.argsort(labels)]
    

#     plt.savefig('figures/path_clustering_binary_n_intersections' 
#                 + str(n_intersections) + '_d_deg' + str(d_deg), dpi=300)
    
    
# cluster_path_intersection_network(d_deg = 1.0, n_intersections=20)

    
def cluster_normalized_path_intersection_network():
    
    C_counts = sparse.load_npz('analysis_output/path_intersection_matrix_counts.npz')
    B = C_counts + C_counts.transpose() - sparse.diags(C_counts.diagonal())
    
    r = np.array(C_counts.diagonal())
    # c =  np.array(sparse.csr_matrix.sum(B, axis=0))[0,:]
    
    R_sqrt_inv = sparse.diags(1./np.sqrt(r)).tocsr()
    # C_sqrt_inv = sparse.diags(1./np.sqrt(c))
    
    C_norm = R_sqrt_inv.dot(B).dot(R_sqrt_inv)
    
    
    A = undirected_network(C_norm)
    A.connected_components()
    
    A_component1 = A.sub_networks[0]
    
    A_component1.compute_laplacian_spectrum()
    plt.plot(A_component1.Lsym_eigenvalues, 'o')
    
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    # lon = data['lon'][A_component1.cluster_indices]
    # lat = data['lat'][A_component1.cluster_indices]
    # time = data['time'][A_component1.cluster_indices]
    # ID = data['ID'][A_component1.cluster_indices]
     
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                           drifter_time = time, drifter_id = ID)
    
    drifter_data.compute_symbolic_sequences(d_deg=1., dt_days = 0.25)
    
    K=10
    A_component1.hierarchical_clustering_ShiMalik(K)
    networks = np.array(A_component1.clustered_networks[K-1])
    labels = [nw.cluster_label for nw in networks]
    networks_sorted = networks[np.argsort(labels)]
    
    gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
    f = plt.figure(constrained_layout=True, figsize = (10,8.5))
    gs = f.add_gridspec(3, 3)
    
    for i in range(0,len(gs_s)):
        g = gs_s[i]
        ax1 = f.add_subplot(gs[g[0], g[1]])
        indices = networks_sorted[i].cluster_indices
        ax1.set_title('Cluster ' + str(i+1) + ' (' + str(len(indices)) + ' trajectories)', size=12)
        drifter_data.trajectories_density(indices=indices)
        drifter_data.plot_discretized_distribution(drifter_data.d_cluster, ax1, logarithmic=True)
        
        
    # ax1 = f.add_subplot(gs[2,0])
    # ax1.plot(range(1, K+1), A_component1.Lsym_eigenvalues[:K], 'o', color = 'maroon')
    # ax1.plot(range(K+1, len(A_component1.Lsym_eigenvalues)+1), A_component1.Lsym_eigenvalues[K:], 'ok', )
    # ax1.set_title('Laplacian eigenvalues', size=12)
    # ax1.grid(True)  
    
    plt.savefig('figures/path_clustering_normalized' + str(n_intersections), dpi=300)
    




"""
Clustering SST
"""

def cluster_SST_data():
    data = np.load('drifter_data_north_atlantic/drifterdata_north_atlantic_withSST.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    SST = data['SST']
    time = data['time']
    ID = data['ID']
        
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_SST = SST, drifter_id = ID)
    
    drifter_data.start_end_times()
    
    plt.hist(drifter_data.trajectory_lengths)
    
    trajectory_length = 365
    max_nans = 300
    indices = np.argwhere(drifter_data.trajectory_lengths>trajectory_length)[:,0]
    indices = [i for i in indices if len(np.argwhere(np.isnan(drifter_data.drifter_SST[i])))<=max_nans]
    
    drifter_data.drifter_longitudes = drifter_data.drifter_longitudes[indices]
    drifter_data.drifter_latitudes = drifter_data.drifter_latitudes[indices]
    drifter_data.drifter_SST = drifter_data.drifter_SST[indices]
    drifter_data.drifter_time = drifter_data.drifter_time[indices]
    drifter_data.drifter_id = drifter_data.drifter_id[indices]
    drifter_data.N = len(drifter_data.drifter_longitudes)
    
    nbins = 40
    drifter_data.compute_SST_histograms(nbins)
    drifter_data.compute_symbolic_sequences(d_deg=1., dt_days = 0.25)
    
    h = drifter_data.SST_histogram_meanzero
    
    for K in range(2,10):
        rs = 0
        kmeans = KMeans(n_clusters=K, random_state=rs).fit(h)        
        labels = kmeans.labels_
        
        
        f = plt.figure(constrained_layout=True, figsize = (10,10))
        gs = f.add_gridspec(3, 3)
        gs_s = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
        
        for l in range(K):
            g = gs_s[l]
            ax1 = f.add_subplot(gs[g[0], g[1]])
            indices = np.argwhere(labels==l)[:,0]
            ax1.set_title('Cluster ' + str(l+1) + ' (' + str(len(indices)) + ' trajectories)', size=12)
            drifter_data.trajectories_density(indices=indices, nmax=trajectory_length * 4)
            drifter_data.plot_discretized_distribution(drifter_data.d_cluster, ax1, logarithmic=True)
            
        plt.savefig('figures/SST_meanzero_clustering_kmeans_nclusters_' + str(K), dpi=300)