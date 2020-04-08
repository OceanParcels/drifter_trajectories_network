import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
import networkx as nx
from create_network import trajectory_data, undirected_network
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture as GMM
# import skfuzzy as fuzz

from c_means_modified import cmeans

drifter_trajectory_data = 'drifter_data_north_atlantic/uniformized_dataset_490days_1deg.npz'

drifter_data = trajectory_data.from_npz(drifter_trajectory_data,  time_interval_days = 1)

# drifter_data.compute_symbolic_sequences(dt_days = 0.25,)

a = 6371.
deg_to_rad = np.pi / 180.

lon = drifter_data.drifter_longitudes
lat = drifter_data.drifter_latitudes

x = a * np.cos(deg_to_rad * lat) * np.sin(deg_to_rad * lon)
y = a * np.cos(deg_to_rad * lat) * np.cos(deg_to_rad * lon)
z = a * np.sin(deg_to_rad * lat)


# x_fplane = np.array([a * lo * deg_to_rad * np.cos(deg_to_rad * la) for lo, la in
#             zip(drifter_data.drifter_longitudes, drifter_data.drifter_latitudes)])
# y_fplane = np.array([a * (la - 45.) *deg_to_rad for la in drifter_data.drifter_latitudes])

X = np.hstack((x, y, z))

fpcs = []
K=2
for K in [2,3,4,5,6,7,8,9,10]:
    cntr, u, u0, d, jm, p, fpc = cmeans(
            X.transpose(), K, 1.5, error=0.005, maxiter=1000, init=None)
    
    fpcs.append(fpc)
    labels = np.argmax(u, axis=0)
    
    # # plt.plot(range(2,11),fpcs)
    
    # # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    # #             X.transpose(), K, 2, error=0.005, maxiter=1000, init=None)
    
    # # Store fpc values for later
    
    
    # Plot assigned clusters, for each data point in training set
        
        
    
    # gmm = GMM(n_components=K).fit(X)
    # labels = gmm.predict(X)
 
    # for K in range(2,15):   
    # kmeans = KMeans(n_clusters=K, random_state=5).fit(X)       
    # labels = kmeans.labels_
        
    f, ax = plt.subplots(figsize=(10,10))
    drifter_data.scatter_initial_position_with_labels(ax, labels = labels)
    ax.set_title(str(K))
    plt.show()


# for l in np.unique(labels):
#     f, ax = plt.subplots(figsize=(10,10))
#     # drifter_data.plot_initial_position_with_labels(ax, labels = labels)
#     drifter_data.trajectories_density(np.argwhere(labels==l)[:,0])
#     drifter_data.plot_discretized_distribution(drifter_data.d_cluster, ax, 
#                                                logarithmic = False)