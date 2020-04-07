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
import skfuzzy as fuzz

data = np.load('drifter_data_north_atlantic/time_coherent_drifters_month_start4length_days_360.npz', allow_pickle=True)
lon = data['lon']
lat = data['lat']
time = data['time']

drifter_data = trajectory_data(drifter_longitudes = lon, drifter_latitudes = lat, 
                                    drifter_time = time, set_nans=True)

drifter_data.drifter_longitudes = drifter_data.drifter_longitudes[:,::4]
drifter_data.drifter_latitudes = drifter_data.drifter_latitudes[:,::4]
drifter_data.drifter_time = drifter_data.drifter_time[:,::4]
drifter_data.N = len(drifter_data.drifter_time)
drifter_data.set_nans()

drifter_data.compute_symbolic_sequences(dt_days = 0.25)

a = 6371.
deg_to_rad = np.pi / 180.
x_fplane = np.array([a * lo * deg_to_rad * np.cos(deg_to_rad * la) for lo, la in
            zip(drifter_data.drifter_longitudes, drifter_data.drifter_latitudes)])
y_fplane = np.array([a * (la - 45.) *deg_to_rad for la in drifter_data.drifter_latitudes])

X = np.hstack((x_fplane, y_fplane))

# fpcs = []
# for K in [2,3,4,5,6,7,8,9,10]:
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#             X.transpose(), K, 2, error=0.005, maxiter=1000, init=None)
    
#     # Store fpc values for later
#     fpcs.append(fpc)

# plt.plot(range(2,11),fpcs)
K=4
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#             X.transpose(), K, 2, error=0.005, maxiter=1000, init=None)

# Store fpc values for later
# fpcs.append(fpc)

# Plot assigned clusters, for each data point in training set
# labels = np.argmax(u, axis=0)
    

gmm = GMM(n_components=K).fit(X)
labels = gmm.predict(X)
    
# kmeans = KMeans(n_clusters=K, random_state=5).fit(X)       
# labels = kmeans.labels_

f, ax = plt.subplots(figsize=(10,10))
drifter_data.plot_initial_position_with_labels(ax, labels = labels)
    
for l in np.unique(labels):
    f, ax = plt.subplots(figsize=(10,10))
    # drifter_data.plot_initial_position_with_labels(ax, labels = labels)
    drifter_data.trajectories_density(np.argwhere(labels==l)[:,0])
    drifter_data.plot_discretized_distribution(drifter_data.d_cluster, ax, 
                                               logarithmic = False)