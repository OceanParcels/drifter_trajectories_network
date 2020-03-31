# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:06:37 2020

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
-----------------
Transition matrix
-----------------
- Advection (to see if it makes sense)
"""

def advection_with_P():
    P = sparse.load_npz('analysis_output/transition_matrix_killed_ddeg05_ddays_5.npz')
    d0 = np.load('analysis_output/initial_distribution_markov_ddeg05_sknorm.npy')
    
    I = np.ones(P.shape[0])
    I[d0==0]=0
    
    I1 = I.copy()
    
    for _ in range(2000):
        # print(np.sum(I1)/np.sum(I))
        I1 = (P.transpose()).dot(I1)
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values()
    
    f = plt.figure(constrained_layout=True, figsize = (15,7))
    gs = f.add_gridspec(1, 1)
    
    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Title', size=16)
    empty_data.plot_discretized_distribution(np.log(I1), ax1, land=False, 
                                             cmap='cividis',  cbar_orientation='vertical')
    

def P_hat_identify_boundaries():
    
    P = sparse.load_npz('analysis_output/transition_matrix_killed_ddeg05_ddays_5.npz')
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
    # P_rn = P_reduced
    # s = np.array(sparse.csr_matrix.sum(P_rn, axis=1))[:,0]
    
    vals, vecs_l = eigs(P_rn.transpose(), k=20, which='LM')
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    vecs_l = vecs_l[:, inds]
    plt.plot(vals, 'o')
    plt.show()
    
    
    pi = np.real(vecs_l[:,0])
    if np.all(pi<=0):
        pi *= -1
    
    PI = sparse.diags(pi)
    # PIinv = sparse.diags(1./pi)
    
    P_rn = P_rn.power(5)
    
    P_W = (PI.dot(P_rn) + P_rn.transpose().dot(PI))/2.
    s = np.array(sparse.csr_matrix.sum(P_W, axis=1))[:,0]
    # S = sparse.diags(1./np.sqrt(s))
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values()
    
    A = undirected_network(P_W, np.array(list(components_sorted[0])), s)
    
    nw1 = A.split_according_to_vn(optimize=True, n=2)
    
    field_plot = np.zeros(P.shape[0])
    field_plot[nw1[0].cluster_indices]= 1
    field_plot[nw1[1].cluster_indices]= 2
    
    f = plt.figure(constrained_layout=True, figsize = (15,7))
    gs = f.add_gridspec(1, 1)
    ax1 = f.add_subplot(gs[0, 0])
    # ax1.set_title('Left eigenvector ' + str(k), size=16)
    empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                              land=True, cmap='cividis', 
                                              cbar_orientation='vertical')
    

def P_hat_hierarchical_spectral_clustering():
    
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
    
    # s = np.array(sparse.csr_matrix.sum(P_rn, axis=1))[:,0]
    
    vals, vecs_l = eigs(P_rn.transpose(), k=10, which='LM')
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    vecs_l = vecs_l[:, inds]
    plt.plot(vals, 'o')
    plt.show()
    
    pi = np.real(vecs_l[:,0])
    if np.all(pi<=0):
        pi *= -1
    
    PI = sparse.diags(pi)
    # PIinv = sparse.diags(1./pi)
    
    P_W = (PI.dot(P_rn) + P_rn.transpose().dot(PI))/2.
    s = np.array(sparse.csr_matrix.sum(P_W, axis=1))[:,0]
    # S = sparse.diags(1./np.sqrt(s))
    
    
    # def drho(mother_network, children_networks):
    #     assert(len(children_networks)==2)
    #     drho = children_networks[0].rho + children_networks[0].rho - mother_network.rho
    #     return drho
    
    networks = {}
    networks[0] = [undirected_network(P_W, np.array(list(components_sorted[0])), s)]
    
    for i in range(1,12):
        
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
            # plt.plot(c_range, drhos)
            # plt.yscale('log')
            # plt.grid(True)
            # plt.title(r'$\Delta \rho_{global}$ for different cutoffs')
            # plt.show()
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
        

# P_hat_hierarchical_spectral_clustering()


# def P_hat_nway_directional_cosine():

P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg05_ddays_5.npz')
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


vals_l, vecs_l = eigs(P_rn.transpose(), k=10, which='LM')
inds = np.argsort(vals_l)[::-1]
vals_l = vals_l[inds]
vecs_l = vecs_l[:, inds]
plt.plot(vals_l, 'o')
plt.show()



"""Plot Eigenvectors directly
"""
vals_r, vecs_r = eigs(P_rn, k=15, which='LM')
inds = np.argsort(vals_r)[::-1]
vals_r = vals_r[inds]
vecs_r = vecs_r[:, inds]


empty_data = trajectory_data()
empty_data.set_discretizing_values()

k_s = range(1,6)
gs_s = [[0,0], [0,1], [1,0], [1,1], [2,0]]
f = plt.figure(constrained_layout=True, figsize = (15,18))
gs = f.add_gridspec(3, 2)

for i in range(len(k_s)):
    g = gs_s[i]
    k = k_s[i]
    ax1 = f.add_subplot(gs[g[0], g[1]])

    field_plot = np.ones(P.shape[0]) *(-10000)
    field_plot[list(components_sorted[0])]= vecs_r[:,k]
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   

    ax1.set_title('n = ' + str(k+1), size=16)
    empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                              land=True, cmap='cividis', 
                                              cbar_orientation='vertical')

ax1 = f.add_subplot(gs[2,1])
ax1.plot(vals_l, 'ok', )
ax1.set_title('Leading eigenvalues')
ax1.grid(True)


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
# P_rn = P_reduced
P_rn =  (P_reduced.transpose().dot(D_inv)).transpose()

# # s = np.array(sparse.csr_matrix.sum(P_rn, axis=1))[:,0]
vals_l, vecs_l = eigs(P_rn.transpose(), k=30, which='LM')
inds = np.argsort(vals_l)[::-1]
vals_l = vals_l[inds]
vecs_l = vecs_l[:, inds]
plt.plot(vals_l, 'o')
plt.show()

vals_r, vecs_r = eigs(P_rn, k=30, which='LM')
inds = np.argsort(vals_r)[::-1]
vals_r = vals_r[inds]
vecs_r = vecs_r[:, inds]

pi = np.real(vecs_l[:,0])
if np.all(pi<=0):
    pi *= -1

PI = sparse.diags(pi)
PIinv = sparse.diags(1./pi)

P_W = (PI.dot(P_rn) + P_rn.transpose().dot(PI))/2.
s = np.array(sparse.csr_matrix.sum(P_W, axis=1))[:,0]
# S = sparse.diags(1./np.sqrt(s))

A = undirected_network(P_W, np.array(list(components_sorted[0])), s)
A.compute_laplacian_spectrum(K=25)

plt.plot(A.eigenvalues, 'o')

K=9
PHI = vecs_r[:,:K]

# for i in range(len(PHI)):
#     PHI[i] /= np.sqrt(PHI[i].dot(PHI[i]))

# X = PHI.dot(PHI.transpose())

kmeans = KMeans(n_clusters=K, random_state=2).fit(PHI)
labels = kmeans.labels_



field_plot = np.zeros(P.shape[0])
field_plot[list(components_sorted[0])]= labels+1
# right_eigenvectors_plot /= np.sum(right_eigenvectors_plot)

empty_data = trajectory_data()
empty_data.set_discretizing_values(d_deg=1.0)

f = plt.figure(constrained_layout=True, figsize = (15,7))
gs = f.add_gridspec(1, 1)
ax1 = f.add_subplot(gs[0, 0])
# ax1.set_title('Left eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')




# # def P_nway_cut()
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
# P_rn = P_reduced
P_rn =  (P_reduced.transpose().dot(D_inv)).transpose()

# # s = np.array(sparse.csr_matrix.sum(P_rn, axis=1))[:,0]

vals, vecs_l = eigs(P_rn.transpose(), k=20, which='LM')
vals, vecs_r = eigs(P_rn, k=20, which='LM')
inds = np.argsort(vals)[::-1]
vals = vals[inds]
vecs_r = vecs_r[:, inds]
plt.plot(vals, 'o')
plt.show()

k=7
kmeans = KMeans(n_clusters=k, random_state=2).fit(np.real(vecs_r[:,:k]))
labels = kmeans.labels_

field_plot = np.zeros(P.shape[0])
field_plot[list(components_sorted[0])]= labels+1
# right_eigenvectors_plot /= np.sum(right_eigenvectors_plot)


empty_data = trajectory_data()
empty_data.set_discretizing_values(d_deg=1.0)

f = plt.figure(constrained_layout=True, figsize = (15,7))
gs = f.add_gridspec(1, 1)
ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('Left eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(field_plot, ax1, logarithmic=False,
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')





# w, v = eigs(S.dot(P_W), k=30, which = 'SM')



# vals_L, vecs_L = eigs(P_hat, k=25, which='LM')
# inds = np.argsort(vals_L)[::-1]
# vals_L = vals_L[inds]
# vecs_L = vecs_L[:, inds]
                
# plt.plot(vals_L, 'o')
# plt.show()

# empty_data = trajectory_data()
# empty_data.set_discretizing_values()


# def ncut(A, indices_1, indices_2):
#     a1 = np.sum(A[indices_1, :][: ,indices_2])
#     a2 = np.sum(A[indices_2, :][: ,indices_1])
    
#     s1 = np.sum(A[indices_1,:])
#     s2 = np.sum(A[indices_2,:])
    
#     return a1/s1 + a2/s2



# k=1

# c_range = np.linspace(np.min(V), np.max(V),100)
# ncuts = []
# for c in c_range:
#     indices_1 = np.argwhere(V<c)[:,0]
#     indices_2 = np.argwhere(V>=c)[:,0]
#     ncuts.append(ncut(P_W, indices_1, indices_2))



# plt.plot(c_range, ncuts)

# kmeans = KMeans(n_clusters=k, random_state=22).fit(np.real(vecs_r[:,:k]))
# labels = kmeans.labels_

right_eigenvectors_plot = np.zeros(P.shape[0])
right_eigenvectors_plot[list(components_sorted[0])]= vecs_L[:,k]
# right_eigenvectors_plot /= np.sum(right_eigenvectors_plot)


f = plt.figure(constrained_layout=True, figsize = (15,7))
gs = f.add_gridspec(1, 1)
ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('Left eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(right_eigenvectors_plot, ax1, logarithmic=False,
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')







vals, vecs_r = eigs(P_rn, k=10, which='LM')

inds = np.argsort(vals)[::-1]
vals = vals[inds]
vecs_r = vecs_r[:, inds]
vecs_l = vecs_l[:, inds]
plt.plot(vals,'o')
plt.show()

empty_data = trajectory_data()
empty_data.set_discretizing_values()

k=3

right_eigenvectors_plot = np.zeros(P.shape[0])
right_eigenvectors_plot[list(components_sorted[0])]=vecs_r[:,k]
right_eigenvectors_plot /= np.sum(right_eigenvectors_plot)
left_eigenvectors_plot = np.zeros(P.shape[0])
left_eigenvectors_plot[list(components_sorted[0])]=vecs_l[:,k]
left_eigenvectors_plot /= np.sum(left_eigenvectors_plot)

f = plt.figure(constrained_layout=True, figsize = (15,7))
gs = f.add_gridspec(1, 2)
ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('Left eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(left_eigenvectors_plot, ax1, logarithmic=False,
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')

ax2 = f.add_subplot(gs[0, 1])
ax2.set_title('Right eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(right_eigenvectors_plot, ax2, 
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')


pi = np.real(vecs_l[:,0])
PI = sparse.diags(pi)
PIinv = sparse.diags(1./pi)

P_hat = (P_rn + (PIinv.dot(P_rn.transpose())).dot(PI))/2.

# piP = PI.dot(P_reduced) 
# P_hat = (piP + piP.transpose())/2.
# n = np.array(sparse.csr_matrix.sum(P_hat, axis=1))[:,0]

# d = np.array(sparse.csr_matrix.sum(P_hat, axis=1))[:,0]
# D_sqrt_inv = sparse.diags(1./np.sqrt(d))
# # D_sqrt = scipy.sparse.diags(np.sqrt(d))
# L = sparse.identity(P_hat.shape[0]) - (D_sqrt_inv.dot(P_hat).dot(D_sqrt_inv))
# print('Computing spectrum')
# vals_L, vecs_L = sparse.linalg.eigsh(L, k=20, which = 'SM')
# inds = np.argsort(vals_L)
# vals_L = vals_L[inds]
# vecs_L = vecs_L[:,inds]
                  
# plt.plot(vals_L, 'o')
# plt.show()

# vecs_L = D_sqrt_inv.dot(vecs_L)


vals_L, vecs_L = eigs(P_hat, k=25, which='LM')
inds = np.argsort(vals_L)[::-1]
vals_L = vals_L[inds]
vecs_L = vecs_L[:, inds]
                
plt.plot(vals_L, 'o')
plt.show()


empty_data = trajectory_data()
empty_data.set_discretizing_values()

k=2

# kmeans = KMeans(n_clusters=k, random_state=22).fit(np.real(vecs_r[:,:k]))
# labels = kmeans.labels_

right_eigenvectors_plot = np.zeros(P.shape[0])
right_eigenvectors_plot[list(components_sorted[0])]= vecs_L[:,k]
right_eigenvectors_plot /= np.sum(right_eigenvectors_plot)


f = plt.figure(constrained_layout=True, figsize = (15,7))
gs = f.add_gridspec(1, 1)
ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('Left eigenvector ' + str(k), size=16)
empty_data.plot_discretized_distribution(right_eigenvectors_plot, ax1, logarithmic=False,
                                          land=True, cmap='cividis', 
                                          cbar_orientation='vertical')






"""
SVD of P
"""

# # def SVD_P(measure='uniform', time_power):

# P = sparse.load_npz('analysis_output/transition_matrix_killed_ddeg05_ddays_5.npz')

# G = nx.from_scipy_sparse_matrix(P, create_using = nx.DiGraph())
# components = np.array(list(nx.strongly_connected_components(G)))
# component_lengths = np.array([len(s) for s in components])
# component_inds = np.argsort(component_lengths)[::-1]
# components_sorted = components[component_inds]
# component_lengths = np.array([len(c) for c in components_sorted])

# P_reduced = P[list(components_sorted[0]),:][:,list(components_sorted[0])]

# empty_data = trajectory_data()
# empty_data.set_discretizing_values()

# # p = np.ones(P_reduced.shape[0])
# p = empty_data.surface_area_field().flatten()[list(components_sorted[0])]
# p /= np.sum(p)
# V = sparse.diags(p)
# VP = (P_reduced.transpose().dot(V)).transpose()

# PI_p =  sparse.diags(np.sqrt(p))
# PI_p_inv =  sparse.diags(1./np.sqrt(p))
# q = P_reduced.transpose().dot(p)
# PI_q_inv =  sparse.diags(1./np.sqrt(q))

# M = (P_reduced.transpose().dot(PI_p))
# M = M.transpose().dot(PI_q_inv)

# from scipy.sparse.linalg import svds
# u, s, vt = svds(M, k=50, which ='LM')

# inds = np.argsort(s)[::-1]
# u = u[:,inds]
# vt = vt[inds,:]
# s = s[inds]

# plt.plot(s,'o')
# plt.show()


# K=4
# x = u[:,:K]
# kmeans = KMeans(n_clusters=K, random_state=0).fit(x)
    
# labels = kmeans.labels_
# eigenvectors_plot = np.zeros(P.shape[0])

# eigenvectors_plot[list(components_sorted[0])]=labels+1



# u2 = u[:,3]
# v2 = vt[3,:]

# u2 = PI_p_inv.dot(u2)
# v2 = PI_q_inv.dot(v2)


# b_range = np.linspace(np.min(v2), np.max(v2), 50)
# b_prime_range = np.linspace(np.min(v2), np.max(v2), 50)

# print(b_range)

# rho_b = []
# bprime_stars = []
# for b in b_range:
#     print('b: ', b)
#     Ib = np.argwhere(u2>b)[:,0]
#     ICb = np.argwhere(u2<=b)[:,0]
    
    
#     dist=[]
#     for b_prime in b_prime_range:
    
#         Jb_prime = np.argwhere(v2>b_prime)[:,0]
#         JCb_prime = np.argwhere(v2<=b_prime)[:,0]
    
#         dist.append(np.abs(np.sum(p[Ib]) - np.sum(q[Jb_prime])))
    
#     b_prime_star = b_prime_range[np.argmin(dist)]
#     bprime_stars.append(b_prime_star)
#     Jb_prime_star = np.argwhere(v2>b_prime_star)[:,0]
#     JCb_prime_star = np.argwhere(v2<=b_prime_star)[:,0]

#     rho_b.append(sparse.csr_matrix.sum(VP[Ib,:][:,Jb_prime_star])/ np.sum(p[Ib]) + \
#         sparse.csr_matrix.sum(VP[ICb,:][:,JCb_prime_star])/ np.sum(p[ICb]))
    
# plt.plot(b_range, rho_b)
# b_star = b_range[np.nanargmax(rho_b)]
# b_prime_star = bprime_stars[np.nanargmax(rho_b)]

# b_star=-0.9
# Ib_star = np.argwhere(u2>b_star)[:,0]
# ICb_star = np.argwhere(u2<=b_star)[:,0]

# V_plot = np.zeros(u2.shape)
# V_plot[Ib_star]=1
# V_plot[ICb_star]=2

# eigenvectors_plot = np.zeros(P.shape[0])
# eigenvectors_plot[list(components_sorted[0])]=V_plot


# f = plt.figure(constrained_layout=True, figsize = (15,7))
# gs = f.add_gridspec(1, 1)

# ax1 = f.add_subplot(gs[0, 0])
# ax1.set_title('Title', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot, ax1, 
#                                          land=True, cmap='cividis', 
#                                          cbar_orientation='vertical')
# plt.show()

# # v = empty_data.surface_area_field().flatten()
# # V = sparse.diags(v)
# # VP = V.dot(P)

# A = undirected_network(P + P.transpose(), np.array(range(P.shape[0])), np.ones(P.shape[0]))

# A.adjacency_matrix.data[:]=1

# A.connected_components()
# A_reduced = A.sub_networks[0]

# d = np.array(sparse.csr_matrix.sum(A_reduced.adjacency_matrix, axis=1))[:,0]
# D = sparse.diags(d)

# L = D - A_reduced.adjacency_matrix

# w, v = sparse.linalg.eigsh(L, k=15, which = 'SM')
# plt.plot(w, 'o')

# # for k in range(20):

# eigenvectors_plot = np.zeros(P.shape[0])
# eigenvectors_plot[list(components_sorted[0])]=v2

# # V = v[:,1]
# # cut = []

# # c_range = np.linspace(np.min(V), np.max(V), 500)
# # for c in c_range:
# #     # inds1 = A_reduced.cluster_indices[]
# #     # inds2 = A_reduced.cluster_indices[]
# #     cut.append(sparse.csr_matrix.sum(A_reduced.adjacency_matrix[V<=c, :][:, V>c]))

# # plt.plot(c_range, cut)

# # c = -0.001
# # eigenvectors_plot = np.zeros(P.shape[0])
# # eigenvectors_plot[A_reduced.cluster_indices]=v[:,1]

# eigenvectors_plot[eigenvectors_plot<c]=-1
# eigenvectors_plot[eigenvectors_plot>=c]=1

    


# time_power = 1
# empty_data = trajectory_data()
# empty_data.set_discretizing_values()

# P = sparse.load_npz('analysis_output/transition_matrix_killed_ddeg05_ddays_5.npz')

# P = P.power(time_power)

# A = P.dot(P.transpose())

# vol = np.array(sparse.csr_matrix.sum(A, axis=1))[:,0]

# A = undirected_network(A, np.array(range(P.shape[0])), vol)
    
# A.connected_components()
# A_reduced = A.sub_networks[0]

# networks = {}
# networks[0] = [A_reduced]

# nw = networks[0]
# nw2 = nw.split_according_to_v2()


# for level in range(1,4):
#     networks_next_level = []
#     print('level ', level)
#     added = 0
#     for nw in networks[level-1]:
        
#         nw2 = nw.split_according_to_v2()
#         if nw.N>100 and nw2[0].rho>ncut_min and nw2[1].rho>ncut_min:
#             networks_next_level += nw2
#             added = 1
#         else:
#             networks_next_level += [nw]
    
#     networks[level] = networks_next_level
    
#     if added==0:
#         break



# A_reduced.compute_laplacian_spectrum()
# plt.plot(A_reduced.eigenvalues, 'o')

# indices_reduced = A_reduced.indices

# c=0

# indices_1 = np.argwhere(A_reduced.eigenvectors[:,1]<c)[:,0]
# indices_2 = np.argwhere(A_reduced.eigenvectors[:,1]>=c)[:,0]

# A1 = adjacency_matrix(A_reduced.A[indices_1,:][:,indices_1], indices = indices_1)
# A2 = adjacency_matrix(A_reduced.A[indices_2,:][:,indices_2], indices = indices_2)

# print(ncut(A_reduced.A, indices_1, indices_2))



# eigenvectors_plot = np.zeros(P.shape[0])
# eigenvectors_plot[A_reduced.indices]=A_reduced.eigenvectors[:,1]
# empty_data = trajectory_data()
# empty_data.set_discretizing_values()

# indices = range()

# c = 0
# eigenvectors_plot[eigenvectors_plot<=0]=-1
# # eigenvectors_plot[eigenvectors_plot>0]=1

# f = plt.figure(constrained_layout=True, figsize = (15,7))
# gs = f.add_gridspec(1, 1)

# ax1 = f.add_subplot(gs[0, 0])
# ax1.set_title('Title', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot, ax1, 
#                                          land=True, cmap='cividis', 
#                                          cbar_orientation='vertical')





def clustering_froyland_2010():
       
    P = sparse.load_npz('analysis_output/transition_matrix_markov_ddeg05_ddays_5.npz')
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values()
    
    p = empty_data.surface_area_field().flatten()
    q = P.transpose().dot(p)
    indices_q = np.argwhere(q!=0)[:,0]
    q = q[indices_q]
    P = P[:,indices_q]
    
    PI_p = sparse.diags(np.sqrt(p))
    PI_q = sparse.diags(1/np.sqrt(q))
    
    T = (P.transpose().dot(PI_p)).transpose().dot(PI_q)
    A = adjacency_matrix(T.dot(T.transpose()))
    A.connected_components()
    A_reduced = A.sub_matrices[0]
    
    w, v = sparse.linalg.eigsh(A_reduced.A, k=10, which = 'LM')
    inds = np.argsort(np.real(w))[::-1]
    v = v[:,inds]
    w = w[inds]
    
    plt.plot(w, 'o')
    
    K = 10
    x_hat = v[:,:K]
    
    x = sparse.diags(1./np.sqrt(p[A_reduced.indices])).dot(x_hat)
    
    kmeans = KMeans(n_clusters=K, random_state=0).fit(x)
    
    labels = kmeans.labels_
    eigenvectors_plot = np.zeros(P.shape[0])
    
    eigenvectors_plot[A_reduced.indices]=labels+1
    
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values()
    
    f = plt.figure(constrained_layout=True, figsize = (15,7))
    gs = f.add_gridspec(1, 1)
    
    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Clustering for 2 clusters', size=16)
    empty_data.plot_discretized_distribution(eigenvectors_plot, ax1, land=False, cmap='cividis', 
                                             cbar_orientation='vertical')
        
        
    
    




# A = adjacency_matrix(T.dot(T.transpose()))
# A.connected_components()
# A_reduced = A.sub_matrices[0]

# A_reduced.compute_laplacian_spectrum()
# plt.plot(A_reduced.eigenvalues, 'o')


# K=4
# rs=0

# X = A_reduced.eigenvectors[:,:K]

# kmeans = KMeans(n_clusters=K, random_state=rs).fit(X)
# labels = kmeans.labels_
# eigenvectors_plot = np.zeros(P.shape[0])

# eigenvectors_plot[A_reduced.indices]=labels+1


# for i in range(15):
#     eigenvectors_plot[:,i][A_reduced.indices] = A_reduced.eigenvectors[:,i]

# empty_data = trajectory_data()
# empty_data.set_discretizing_values()

# f = plt.figure(constrained_layout=True, figsize = (15,7))
# gs = f.add_gridspec(1, 1)

# ax1 = f.add_subplot(gs[0, 0])
# ax1.set_title('Title', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot, ax1, land=False, cmap='cividis', 
#                                          cbar_orientation='vertical')









# v = empty_data.surface_area_field().flatten()
# V = sparse.diags(np.sqrt(v))

# # Vinv = sparse.diags(1./v)
# P = sparse.load_npz('analysis_output/transition_matrix_killed_ddeg05_ddays_5.npz')
# Vinv = sparse.diags(1./np.sqrt(P.transpose().dot(v)))

# VP = (P.transpose().dot(V)).transpose().dot(Vinv)

# A = adjacency_matrix(VP.dot(VP.transpose()))
# A.connected_components()
# A_reduced = A.sub_matrices[0]

# A_reduced.compute_laplacian_spectrum()
# plt.plot(A_reduced.eigenvalues, 'o')


# K=7
# rs=0

# X = A_reduced.eigenvectors[:,:K]

# kmeans = KMeans(n_clusters=K, random_state=rs).fit(X)
# labels = kmeans.labels_
# eigenvectors_plot = np.zeros(P.shape[0])

# eigenvectors_plot[A_reduced.indices]=labels+1


# # for i in range(15):
# #     eigenvectors_plot[:,i][A_reduced.indices] = A_reduced.eigenvectors[:,i]

# empty_data = trajectory_data()
# empty_data.set_discretizing_values()

# f = plt.figure(constrained_layout=True, figsize = (15,7))
# gs = f.add_gridspec(1, 1)

# ax1 = f.add_subplot(gs[0, 0])
# ax1.set_title('Title', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot, ax1, land=False, cmap='cividis', 
#                                          cbar_orientation='vertical')






# ax2 = f.add_subplot(gs[0, 1])
# ax2.set_title('Out degree', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot[:,1], ax2, land=True, cmap='cividis')


# ax2 = f.add_subplot(gs[1, 0])
# ax2.set_title('Out degree', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot[:,2], ax2, land=True, cmap='cividis')


# ax2 = f.add_subplot(gs[1, 1])
# ax2.set_title('Out degree', size=16)
# empty_data.plot_discretized_distribution(eigenvectors_plot[:,3], ax2, land=True, cmap='cividis')












"""
Local entropy transition matrix
"""

def entropy_and_outdegree():
    P = sparse.load_npz('analysis_output/transition_matrix.npz')
    
    E = P.copy()
    N = P.copy()
    
    E.data = np.multiply(E.data, np.log(E.data))
    N.data[:] = 1
    
    out_degree = np.array(sparse.csr_matrix.sum(N, axis=1))[:,0]
    
    e = np.array(sparse.csr_matrix.sum(E, axis=1))[:,0]
    
    empty_data = trajectory_data()
    empty_data.set_discretizing_values()
    
    f = plt.figure(constrained_layout=True, figsize = (16,6))
    gs = f.add_gridspec(1, 4)
    
    ax1 = f.add_subplot(gs[0, :2])
    ax1.set_title('Transition matrix outgoing entropy', size=16)
    empty_data.plot_discretized_distribution(e, ax1, land=True, cmap='cividis')
    
    ax2 = f.add_subplot(gs[0, 2:])
    ax2.set_title('Out degree', size=16)
    empty_data.plot_discretized_distribution(out_degree, ax2, land=True, cmap='cividis')



"""
Temperature network
"""
def SST_clutering():

    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic_withSST.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    SST = data['SST']
    time = data['time']
    
    indices = [i  for i in range(len(SST)) if len(SST[i][SST[i]<90])>1460]
    
    lon = lon[indices]
    lat = lat[indices]
    SST = SST[indices]
    time = time[indices]
    
    SST_anomaly = np.array([SST[i][SST[i]<90] - np.mean(SST[i][SST[i]<90]) for i in range(len(SST))])
    
    nbins = 40
    """
    For SST anomaly
    """
    
    SST_min_anomaly = np.min(np.array([np.min(s) for s in SST_anomaly]))
    SST_max_anomaly = np.max(np.array([np.max(s[s<90]) for s in SST_anomaly]))
    sst_hist_anomaly = np.array([np.histogram(SST_anomaly[i], bins=nbins, density=True, range=(SST_min_anomaly, SST_max_anomaly))[0] for i in range(len(SST_anomaly))])
    x_range_SSTA = np.linspace(SST_min_anomaly, SST_max_anomaly, nbins)
    
    """
    For SST
    """
    
    SST_min = np.min(np.array([np.min(s) for s in SST]))
    SST_max = np.max(np.array([np.max(s[s<90]) for s in SST]))
    sst_hist = np.array([np.histogram(SST[i][SST[i]<90], bins=nbins, density=True, range=(SST_min, SST_max))[0] for i in range(len(SST))])
    x_range_SST = np.linspace(SST_min, SST_max, nbins)
    sst_hist_mean = np.mean(sst_hist, axis=1)
    sst_hist_zeromean = np.array([sst_hist[i] - sst_hist_mean[i] for i in range(len(sst_hist))])
    
    plt.plot(x_range_SST, sst_hist[200], 'o')
    plt.show()
    
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=8)
    # #X_new = pca.fit_transform(T_datamatrix_zeromean)
    # X_new = pca.fit_transform(sst_hist_zeromean)
    
    
    from sklearn.mixture import GaussianMixture as GMM
    bic_full = []
    bic_tied = []
    bic_diag = []
    bic_spherical = []
    n_components = range(2,20)
    
    for n in n_components:
        
        gmm_full = GMM(n_components=n, covariance_type = 'full').fit(sst_hist_anomaly)
        gmm_tied = GMM(n_components=n, covariance_type = 'tied').fit(sst_hist_anomaly)
        gmm_diag = GMM(n_components=n, covariance_type = 'diag').fit(sst_hist_anomaly)
        gmm_spherical = GMM(n_components=n, covariance_type = 'spherical').fit(sst_hist_anomaly)
        # labels = gmm.predict(sst_hist_anomaly)
        bic_full.append(gmm_full.bic(sst_hist_anomaly))
        bic_tied.append(gmm_tied.bic(sst_hist_anomaly))
        bic_diag.append(gmm_diag.bic(sst_hist_anomaly))
        bic_spherical.append(gmm_spherical.bic(sst_hist_anomaly))
    
    plt.plot(n_components, bic_full, label='full')
    plt.plot(n_components, bic_tied, label='tied')
    plt.plot(n_components, bic_diag, label='diag')
    plt.plot(n_components, bic_spherical, label='spherical')
    plt.legend()
    plt.xlabel('# Components')
    plt.ylabel('BIC')
    
    print('Optimal number of components: ', n_components[np.argmin(bic_full)])
    
    # gmm = GMM(n_components=4).fit(sst_hist_zeromean)
    # labels = gmm.predict(sst_hist_zeromean)
    
    kmeans = KMeans(n_clusters=4, random_state=0).fit(sst_hist_zeromean)
    labels = kmeans.labels_
    
    indices = np.random.randint(0, len(lon), size=400)
    labels = labels[indices]
    
    lon_reduced = lon[indices]
    lat_reduced = lat[indices]
    time_reduced = time[indices]
    
    lonplot = lon_reduced[0]
    latplot = lat_reduced[0]
    timeplot = time_reduced[0]
    c = np.array([labels[0]] * len(lon_reduced[0]))
    
    for i in range(1, len(lon_reduced)):
        lonplot = np.append(lonplot, lon_reduced[i])
        latplot = np.append(latplot, lat_reduced[i])
        timeplot = np.append(timeplot, time_reduced[i])
        c = np.append(c, np.array([labels[i]] * len(lon_reduced[i])))
    
    plt.figure(figsize = (7,7))
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    xs, ys = m(lonplot, latplot)
    m.scatter(xs, ys, s=2, c=c,  alpha = 0.1)
    plt.colorbar(shrink = 0.6)
    
    for l_select in range(4):
        plt.figure(figsize = (7,7))
        plt.title('Number component: ' + str(l_select))
        m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
        m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
        m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
        m.drawcoastlines()
        xs, ys = m(lonplot, latplot)
        m.scatter(xs[c==l_select], ys[c==l_select], s=2, c=c[c==l_select],  alpha = 0.1)
        plt.colorbar(shrink = 0.6) 
        plt.show()
        t = timeplot[c==l_select]
        plt.hist(t, bins=50)
        plt.show()


"""
---------------------------
Trajectory network
---------------------------
"""

def find_multiple_clusters():
    
    filename ="analysis_output/intersection_network_spectrum_overlaps_n5.npz"
    # filename ="analysis_output/intersection_network_normalized_spectrum.npz"
    data = np.load(filename, allow_pickle = True)
    
    vals = data['vals']
    vecs = data['vecs']
    
    lon = data['lon']
    lat = data['lat']
    
    plt.plot(vals, 'o')
    
    for n_vecs in [2,3,4,5,6,7,8,9,10]:
        n_clusters = n_vecs
        data_lowdim = vecs[:,:n_vecs]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(data_lowdim))
        labels = kmeans.labels_
        
        # V[V<0]=-1
        # V[V>=0]=1
        
        indices = np.random.randint(0, len(lon), size=400)
        labels = labels[indices]
        # V = vecs[indices,7]
        lon_reduced = lon[indices]
        lat_reduced = lat[indices]
        
        lonplot = lon_reduced[0]
        latplot = lat_reduced[0]
        c = np.array([labels[0]] * len(lon_reduced[0]))
        
        for i in range(1, len(lon_reduced)):
            lonplot = np.append(lonplot, lon_reduced[i])
            latplot = np.append(latplot, lat_reduced[i])
            c = np.append(c, np.array([labels[i]] * len(lon_reduced[i])))
        
        plt.figure(figsize = (7,7))
        plt.title("Binary network, clustering for n = " + str(n_vecs))
        m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
        m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
        m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
        m.drawcoastlines()
        xs, ys = m(lonplot, latplot)
        m.scatter(xs, ys, s=2, c=c,  alpha = 0.1)
        plt.colorbar(shrink = 0.6) 
        


def plot_trajectory_spectrum(filename, k):
    
    data = np.load(filename, allow_pickle = True)
    
    vals = data['vals']
    vecs = data['vecs']
    
    lon = data['lon']
    lat = data['lat']
    
    plt.plot(vals, 'o')
    
    indices = np.random.randint(0, len(lon), size=400)
    # V = vecs[indices,7]
    lon_reduced = lon[indices]
    lat_reduced = lat[indices]
     
    V=vecs[indices,k]
    
    lonplot = lon_reduced[0]
    latplot = lat_reduced[0]
    c = np.array([V[0]] * len(lon_reduced[0]))
    
    for i in range(1, len(lon_reduced)):
        lonplot = np.append(lonplot, lon_reduced[i])
        latplot = np.append(latplot, lat_reduced[i])
        c = np.append(c, np.array([V[i]] * len(lat_reduced[i])))
    
    plt.figure(figsize = (7,7))
    plt.title("Second eigenvector")
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    plt.title(str(k))
    xs, ys = m(lonplot, latplot)
    m.scatter(xs, ys, s=2, c=c,  alpha = 0.1)
    plt.colorbar(shrink = 0.6) 
        
# for k in [1,2,3,4,5,6,7,8,9,10]:    
#     plot_trajectory_spectrum(filename ="analysis_output/intersection_network_spectrum_overlaps_n1.npz", k=k)


def plot_degree():
    W = sparse.load_npz("analysis_output/trajectory_network.npz")
    W = W + W.transpose()
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    
    lon0 = [lon[i][0] for i in range(len(lon))]
    lat0 = [lat[i][0] for i in range(len(lat))]
        
    d = np.array(sparse.csr_matrix.sum(W, axis=1))[:,0]
    
    
    plt.figure(figsize = (7,7))
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    xs, ys = m(lon0, lat0)
    m.scatter(xs, ys, s=2, c=d,  alpha = 1)
    plt.colorbar(shrink = 0.6) 
    
    
    



def normalize_network():
    W = sparse.load_npz("analysis_output/trajectory_network.npz")
    W = W + W.transpose() - sparse.diags(sparse.csr_matrix.diagonal(W))
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    description = "trajectory based network, not time info"
    drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, drifter_id = ID, 
                                    data_description = description)
    drifter_data.compute_symbolic_sequences(d_deg = 0.1, dt_days = 0.25)
    
    sequence = drifter_data.symbolic_sequence
    
    trajectory_length = []
    for i in range(len(sequence)):
        trajectory_length.append(len(np.unique(sequence[i])))
    
    weight_matrix = np.array([[1./(np.sqrt(trajectory_length[i]) * np.sqrt(trajectory_length[j])) for i in range(len(trajectory_length))] 
                              for j in range(len(trajectory_length))])
    
    weight_matrix = sparse.csr_matrix(weight_matrix)
    
    normalized_network = sparse.csr_matrix.multiply(weight_matrix, W)
    
    plt.hist(normalized_network.data[normalized_network.data<100], bins=100)
    
    sparse.save_npz('trajectory_network_normalized', normalized_network)


def compute_spectrum(filename):
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    
    W = sparse.load_npz(filename)
    
    G = nx.from_scipy_sparse_matrix(W, create_using = nx.Graph())
    components = np.array(list(nx.connected_components(G)))
    component_lengths = np.array([len(s) for s in components])
    component_inds = np.argsort(component_lengths)[::-1]
    components_sorted = components[component_inds]
    largest_connected_component = list(components_sorted[0])
    
    W = W[largest_connected_component,:][:,largest_connected_component]
    
    d = np.array(sparse.csr_matrix.sum(W, axis=1))[:,0]
    Dinv = sparse.diags(1./d)
    L = Dinv.dot(W)
    vals, vecs = eigs(L, k=20, which = 'LM')
    inds = np.argsort(vals)[::-1]
    vals=vals[inds]
    vecs=vecs[:,inds]
    plt.plot(vals,'o')
    plt.show()
        
    lon = lon[largest_connected_component]
    lat = lat[largest_connected_component]
    np.savez('intersection_network_normalized_spectrum', vals=vals, vecs=vecs, lon=lon, lat=lat)

# compute_spectrum('analysis_output/trajectory_network_normalized.npz')
    

def plot_distance_data():
    C = np.load("analysis_output/mixing_network_first_try.npz")["count_matrix"]
    c_data = C[C!=0].flatten()
    
    h, x = np.histogram(c_data[~np.isnan(c_data)], bins=99)
        
    x_plot = x[0] + np.cumsum(np.diff(x))
    plt.loglog(x_plot, h, 'o')
    


def plot_intersections_histogram():
    W = sparse.load_npz("analysis_output/trajectory_network.npz")
    W = W + W.transpose()
    


    h, x = np.histogram(W.data[W.data<100], bins=99)
    
    x_plot = x[0] + np.cumsum(np.diff(x))
    plt.plot(x_plot, h, 'o')
    
    plt.xlabel('# intersections')

# plot_intersections_histogram()

def compute_spectra(min_overlaps=1):

    A = sparse.load_npz("analysis_output/trajectory_network.npz").toarray()
    A = A + A.transpose() - np.diag(np.diagonal(A))
    
    # W = W + W.transpose() - sparse.diags(sparse.csr_matrix.diagonal(W))
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    
    A[A<min_overlaps] = 0
    A[A>=min_overlaps] = 1
    # W.data[W.data<min_overlaps] = 0
    # W.data[W.data>=min_overlaps] = 1
    
    W = sparse.csr_matrix(A)
    
    G = nx.from_scipy_sparse_matrix(W, create_using = nx.Graph())
    components = np.array(list(nx.connected_components(G)))
    component_lengths = np.array([len(s) for s in components])
    component_inds = np.argsort(component_lengths)[::-1]
    components_sorted = components[component_inds]
    largest_connected_component = list(components_sorted[0])
    
    W = W[largest_connected_component,:][:,largest_connected_component]
    
    d = np.array(sparse.csr_matrix.sum(W, axis=1))[:,0]
    # Dinv= np.diag(1./d)
    Dinv = sparse.diags(1./d)
    L = Dinv.dot(W)
    # Lsparse = sparse.csr_matrix(L)
    vals, vecs = eigs(L, k=20, which = 'LM')
    inds = np.argsort(vals)[::-1]
    vals=vals[inds]
    vecs=vecs[:,inds]
    plt.plot(vals,'o')
    plt.show()
    
    lon = lon[largest_connected_component]
    lat = lat[largest_connected_component]
    
    np.savez('intersection_network_spectrum_overlaps_n' + str(min_overlaps), vals=vals, vecs=vecs, lon=lon, lat=lat)

# compute_spectra(30)
# for min_overlaps in [1,2,3,4,5,6,7,8,9,10]:
#     compute_spectra(min_overlaps)


# k=5
# V=vecs[:,k]

# # lonplot = lon[0]
# # latplot = lat[0]
# c = np.array([V[0]] * len(lon[0]))

# for i in range(1, len(lon)):
#     # lonplot = np.append(lonplot, lon[i])
#     # latplot = np.append(latplot, lat[i])
#     c = np.append(c, np.array([V[i]] * len(lon[i])))

# plt.figure(figsize = (7,7))
# plt.title("Second eigenvector")
# m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
# m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
# m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
# m.drawcoastlines()
# plt.title(str(k))
# xs, ys = m(lonplot, latplot)
# m.scatter(xs, ys, s=2, c=c,  alpha = 0.1)
# plt.colorbar(shrink = 0.6) 



"""
---------------------------
Mixing network
---------------------------
"""


"""
Construct network with cutoff on count matrix
"""

def local_network_properties():
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    C = np.load("analysis_output/mixing_network_first_try.npz")["count_matrix"]
    
    A = np.zeros(C.shape)
    A[C<15]=1
    
    degree = np.sum(A, axis=1)
    
    lonplot = lon[0]
    latplot = lat[0]
    c = np.array([degree[0]] * len(lonplot))
    
    for i in range(1, len(lon)):
        lonplot = np.append(lonplot, lon[i])
        latplot = np.append(latplot, lat[i])
        c = np.append(c, np.array([degree[i]] * len(lon[i])))
    
    plt.figure(figsize = (7,7))
    plt.title("Second eigenvector")
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    
    xs, ys = m(lonplot, latplot)
    m.scatter(xs, ys, s=2, c=np.log(c),  alpha = 0.1)
    # m.scatter(xs, ys, s=2, alpha = 0.1)
    plt.colorbar(shrink = 0.6)
    

def connected_component_and_spectrum():

    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lon = data['lon']
    lat = data['lat']
    time = data['time']
    ID = data['ID']
    
    C = np.load("analysis_output/mixing_network_first_try.npz")["count_matrix"]
    
    A = np.zeros(C.shape)
    A[C<15]=1
    
    G = nx.from_numpy_matrix(A, create_using = nx.Graph())
    components = np.array(list(nx.connected_components(G)))
    component_lengths = np.array([len(s) for s in components])
    component_inds = np.argsort(component_lengths)[::-1]
    components_sorted = components[component_inds]
    
    for c in components_sorted[:10]:
        lon_plot = lon[list(c)][0]
        lat_plot = lat[list(c)][0]
        
        for i in range(1, len(lon[list(c)])):
            lon_plot = np.append(lon_plot, lon[list(c)][i])
            lat_plot = np.append(lat_plot, lat[list(c)][i])
    
        plt.figure(figsize = (7,7))
        plt.title("Connected component size: " + str(len(c)))
        m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
        m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
        m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
        m.drawcoastlines()
        xs, ys = m(lon_plot, lat_plot)  
        m.scatter(xs, ys, s=2, alpha = 0.1)
        
    
    largest_cc = list(components_sorted[0])
    
    A_red = A[largest_cc,:][:,largest_cc]
    lon_red = lon[largest_cc]
    lat_red = lat[largest_cc]
    
    d = np.sum(A_red, axis=1)
    D = np.diag(d)
    Dinv = np.diag(1./d)
    L = Dinv.dot(A_red)
    Lsparse = sparse.csr_matrix(L)
    vals, vecs = eigs(Lsparse, k=15, which = 'LM')
    inds = np.argsort(vals)[::-1]
    vals=vals[inds]
    vecs=vecs[:,inds]
    plt.plot(vals,'o')


"""
Plot second eigenvector
"""
def plot_second_eigenvector():

    V = vecs[:,1]
    # c=-0.029
    
    lonplot = lon_red[0]
    latplot = lat_red[0]
    c = np.array([V[0]] * len(lon_red[0]))
    
    for i in range(1, len(lon_red)):
        lonplot = np.append(lonplot, lon_red[i])
        latplot = np.append(latplot, lat_red[i])
        c = np.append(c, np.array([V[i]] * len(lon_red[i])))
    
    plt.figure(figsize = (7,7))
    plt.title("Second eigenvector")
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    
    xs, ys = m(lonplot, latplot)
    m.scatter(xs, ys, s=2, c=c,  alpha = 0.1)
    # m.scatter(xs, ys, s=2, alpha = 0.1)
    plt.colorbar(shrink = 0.6)



"""
Spectral clustering
"""
def spectral_clustering_mixing_network():

    n_vecs = 6
    n_clusters = 4
    data_lowdim = vecs[:,1:n_vecs+1]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(data_lowdim))
    labels = kmeans.labels_
    
    # lon0 = np.array([lon_red[i][0] for i in range(len(lat_red))])
    # lat0 = np.array([lat_red[i][0] for i in range(len(lat_red))])
    
    plt.figure(figsize = (7,7))
    plt.title("Nclusters = " + str(n_clusters))
    m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    m.drawcoastlines()
    
    for l in np.unique(labels):
        
        lonplot = lon_red[labels == l][0]
        latplot = lat_red[labels == l][0]
        
        for i in range(1, len(lon_red[labels == l])):
            lonplot = np.append(lonplot, lon_red[labels == l][i])
            latplot = np.append(latplot, lat_red[labels == l][i])
        
        # lonplot = lon0[labels == l]
        # latplot = lat0[labels == l]
        xs, ys = m(lonplot, latplot)
        m.scatter(xs, ys, s=2, alpha = 0.1)
    
    # plt.show()



"""
Transition matrix
"""

def compute_volume_flow():
    empty_data = trajectory_data(drifter_longitudes = [], 
                                  drifter_latitudes = [], 
                                  drifter_time = [], drifter_id = [])
    empty_data.set_discretizing_values()
    
    P = sparse.load_npz('transition_matrix.npz')
    d0 = np.load('initial_distribution.npy')[:,0]
    I=np.zeros(d0.shape)
    I[d0>0]=1
    empty_data.plot_discretized_distribution(I)
    projector = sparse.diags(I)
    empty_data.plot_discretized_distribution(I)

    #volume flow from subtropical to subpolar gyre
    V = P.transpose().dot(sparse.diags(empty_data.volume_field().flatten()))
    subpolar = np.zeros(empty_data.n_horizontal)
    subtropical = np.zeros(empty_data.n_horizontal)
    i0 = empty_data.coords_to_matrixindex2D((-120.,50))
    subpolar[i0:] = 1
    subtropical[0:i0]=1
    # subpolar = projector.dot(subpolar)
    suptropical = projector.dot(subtropical)
    # suptropical_projector = sparse.diags(suptropical)
    suptropical /= np.sum(suptropical)
    empty_data.plot_discretized_distribution(suptropical)
    
    V_reduced = np.zeros((2,2))
    V_reduced[0,0] = (V.transpose().dot(subtropical)).dot(subtropical)
    V_reduced[0,1] = (V.transpose().dot(subtropical)).dot(subpolar)
    V_reduced[1,0] = (V.transpose().dot(subpolar)).dot(subtropical)
    V_reduced[1,1] = (V.transpose().dot(subpolar)).dot(subpolar)

    print('Sv / m transport: ', V_reduced[0,1] / (1e6 * 5 * 86400))


def likelyhood_particle_stay():
    empty_data = trajectory_data(drifter_longitudes = [], 
                                  drifter_latitudes = [], 
                                  drifter_time = [], drifter_id = [])
    empty_data.set_discretizing_values()
    P = sparse.load_npz('transition_matrix.npz')
    
    subtropical = np.zeros(empty_data.n_horizontal)
    subpolar_cut_index = empty_data.coords_to_matrixindex2D((-120.,50))
    subtropical[0:subpolar_cut_index]=1
    
    data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
    lat = data['lat']
    lon = data['lon']
    
    lat0 = np.array([lat[i][0] for i in range(len(lat)) if lat[i][0] < 50.])
    lon0 = np.array([lon[i][0] for i in range(len(lat)) if lat[i][0] < 50.])
    livetimes = np.array([len(lat[i])/(4 * 5) for i in range(len(lat))  if lat[i][0] < 50]) # in 5 days
    
    p_stay = np.empty(len(lat0))
    
    for i in range(len(lat0)):
        if i%50 == 0 and i>0:
            # h, x = np.histogram(p_stay[p_stay<2], bins=200)
            # plt.plot(x[1:], h, 'o')
            # plt.xlim([0.1,1.1])
            # plt.show()
            print(str(i) + ' / ' + str(len(lat0)))
        
        index_0 = empty_data.coords_to_matrixindex2D((lon0[i],lat0[i]))
        v = np.zeros(empty_data.n_horizontal)
        v[index_0] = 1
        t = int(livetimes[i])
        
        for _ in range(t):
            v = P.transpose().dot(v)
        
        v /= np.sum(v)
        p_stay[i] = subtropical.dot(v)
    
    return p_stay

    plt.hist(p_stay, bins=100)
    
    p_mean = np.nanmean(p_stay)
    p_sigma = np.nanstd(p_stay)
    
    from scipy.stats import binom
    n, p = 4074, 1-p_mean
    
    x = np.arange(1,200,1)
    plt.plot(x, binom.pmf(x, n, p))


# def cluster_transition_matrix():


# empty_data.plot_discretized_distribution(x_full, land=True, title = 'clustering')







# def random_walker():
# empty_data = trajectory_data(drifter_longitudes = [], 
#                               drifter_latitudes = [], 
#                               drifter_time = [], drifter_id = [])
# empty_data.set_discretizing_values()
# P = sparse.load_npz('transition_matrix.npz')

# data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
# lat = data['lat']
# lon = data['lon']

# lat0 = np.array([lat[i][0] for i in range(len(lat)) if lat[i][0] < 50.])
# lon0 = np.array([lon[i][0] for i in range(len(lat)) if lat[i][0] < 50.])

# i_cut = empty_data.coords_to_matrixindex2D((-120.,50))

# index_0 = empty_data.coords_to_matrixindex2D((lon0[0], lat0[0]))

# p0 = index_0

# path = [index_0]

# for _ in range(3 * 69):
#     path += [np.random.choice(range(empty_data.n_horizontal), p=P[path[-1],:].toarray()[0])]

# path = np.array(path) 

# print(np.any(path>i_cut))
    
    
    
    
    
    
    

# v = suptropical.copy()
# empty_data.plot_discretized_distribution(v)

# #Long-term crossings of uniform drifter distribution

# # for _ in range(69):
# #     v = P.transpose().dot(v)
# #     v /= np.sum(v)

# # v = suptropical_projector.dot(v)
# # empty_data.plot_discretized_distribution(v)

# shares = []

# for j in range(500):
#     if j%20 ==0:
#         print(j)
#     v = suptropical.copy()
#     for _ in range(j):
#         v = P.transpose().dot(v)
#         v /= np.sum(v)

#     shares += [v.dot(subpolar)]

# plt.plot(np.array(list(range(500)))/69, shares)
# plt.xlabel('years')

# data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
# lat = data['lat']
# lon = data['lon']

# indices = [empty_data.coords_to_matrixindex2D((lo[0], la[0])) for lo, la in zip(lon, lat) if la[0]<50]
# unique_indices, counts = np.unique(indices, return_counts = True)


# initial_drifters = np.zeros(empty_data.n_horizontal)
# initial_drifters[unique_indices] = counts

# empty_data.plot_discretized_distribution(initial_drifters)

# drifter_distr = initial_drifters.copy()
# drifter_distr /= np.sum(drifter_distr)

# for _ in range(69):
#         drifter_distr = P.transpose().dot(drifter_distr)
#         drifter_distr /= np.sum(drifter_distr)

# empty_data.plot_discretized_distribution(drifter_distr)

# print(drifter_distr.dot(subpolar) *np.sum(initial_drifters) )

# lat0 = np.array([lat[i][0] for i in range(len(lat))])
# livetimes = np.array([len(lat[i])/(4 * 365) for i in range(len(lat))])
# plt.hist(livetimes)
# len(lat0[lat0<50])


# # empty_data.plot_discretized_distribution(np.log(v/np.max(v)))


# # empty_data.plot_discretized_distribution(np.log(v))








