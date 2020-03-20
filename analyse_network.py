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

"""
Mixing network
"""


data = np.load('trajectory_data_north_atlantic/drifterdata_north_atlantic.npz', allow_pickle=True)
lon = data['lon']
lat = data['lat']
time = data['time']
ID = data['ID']

C = np.load("mixing_network_first_try.npz")["count_matrix"]

A = np.zeros(C.shape)
A[C<10]=1

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
    
    
print('Network is set up')
if not nx.is_connected(G):
    print('Graph is not connected. Finding largest connected component.')
    largest_cc = np.array(list(max(nx.connected_components(G),
                                   key=len)))
else:
    print('Graph is connected.')
    largest_cc = np.array(range(A.shape[0]))

np.save('largest_connected_component', largest_cc) 

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



plt.figure(figsize = (7,7))
plt.title(l)
m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
m.drawcoastlines()

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





n_vecs = 6
n_clusters = 10
data_lowdim = vecs[:,1:n_vecs+1]

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(data_lowdim))
labels = kmeans.labels_

lon0 = np.array([lon_red[i][0] for i in range(len(lat_red))])
lat0 = np.array([lat_red[i][0] for i in range(len(lat_red))])

plt.figure(figsize = (7,7))
plt.title("Nclusters = " + str(n_clusters))
m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
m.drawcoastlines()

for l in np.unique(labels):

    # plt.figure(figsize = (7,7))
    # plt.title(l)
    # m = Basemap(projection='mill',llcrnrlat=-10,urcrnrlat=85, llcrnrlon=-110,urcrnrlon=30,resolution='c')
    # m.drawparallels([0,35,70], labels=[True, False, False, True], linewidth=1.8, size=15)
    # m.drawmeridians([-80,-35,10], labels=[False, False, False, True], linewidth=1.8, size=15)
    # m.drawcoastlines()
    
    lonplot = lon_red[labels == l][0]
    latplot = lat_red[labels == l][0]
    
    for i in range(1, len(lon_red[labels == l])):
        lonplot = np.append(lonplot, lon_red[labels == l][i])
        latplot = np.append(latplot, lat_red[labels == l][i])
    
    # lonplot = lon0[labels == l]
    # latplot = lat0[labels == l]
    xs, ys = m(lonplot, latplot)
    m.scatter(xs, ys, s=2, alpha = 0.1)

plt.show()



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


def cluster_transition_matrix():

    empty_data = trajectory_data(drifter_longitudes = [], 
                               drifter_latitudes = [], 
                               drifter_time = [], drifter_id = [])
    empty_data.set_discretizing_values()
    
     
    P = sparse.load_npz('transition_matrix.npz')
    d0 = np.load('initial_distribution.npy')[:,0]
    indices = np.argwhere(d0>0)[:,0]
    
    P = P[indices,:][:,indices]
    
    # I=np.zeros(d0.shape)
    # I[d0>0]=1
    
    p = empty_data.volume_field().flatten()
    p = p[indices]
    q = P.transpose().dot(p)
    
    indices_q = np.argwhere(q!=0)[:,0]
    q = q[indices_q]
    P = P[:,indices_q]
    
    PI_p = sparse.diags(np.sqrt(p))
    PI_q = sparse.diags(1/np.sqrt(q))
    
    T = (P.transpose().dot(PI_p)).transpose().dot(PI_q)
    
    # empty_data.plot_discretized_distribution(I)
    # projector = sparse.diags(I)
    # empty_data.plot_discretized_distribution(I)
    
    #volume flow from subtropical to subpolar gyre
    # V = P.transpose().dot(sparse.diags(empty_data.volume_field().flatten()))
    
    from scipy.sparse.linalg import svds
    
    u, s, vt = svds(T, k=10)
    
    # print(s[l])
    
    x_hat = u[:,:2]
    
    x = sparse.diags(1./np.sqrt(p)).dot(x_hat)
    
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
    labels = kmeans.labels_
    
    x_full = np.zeros(empty_data.n_horizontal)
    x_full[indices] = labels + 1
    
    # v_plot = np.zeros(empty_data.n_horizontal)
    # v_plot[(x_full<c) & (x_full !=0)]=-1
    # v_plot[x_full>c]=1
    
    
    empty_data.plot_discretized_distribution(x_full, land=True, title = 'clustering')







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



