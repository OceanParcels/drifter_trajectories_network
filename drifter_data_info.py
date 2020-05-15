"""
Detecting flow features in scarce trajectory data using networks derived from 
symbolic itineraries: an application to surface drifters in the North Atlantic
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

"""

"""
Script to generate drifter info (Fig 1 in the paper).

Requires: "drifterdata_north_atlantic.npz", created with "constrain_drifterdata_to_northatlantic.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data
import matplotlib.colors as colors

d_deg=[2.0]
domain_edges = (-100., 80, 0, 85) # full domain, to cover all points along any trajectory    
drifter_data = trajectory_data.from_npz('drifterdata_north_atlantic.npz',
                                          time_interval=0.25, n_step = 1, domain_type = "north_atlantic_domain")

#Total counts (fig. 1a)
drifter_data.compute_symbolic_itineraries(bin_size = d_deg, dt = 0.25)
sequence = drifter_data.symbolic_sequence
total_symbols = []
for s in sequence: total_symbols += list(s[:])
total_symbols = np.array(total_symbols).astype(int)
total_symbols = total_symbols[total_symbols>=0]
unique_symbols, counts = np.unique(total_symbols, return_counts=True)
drifter_distribution = np.zeros(drifter_data.n_horizontal)
drifter_distribution[unique_symbols] = counts
drifter_distribution = np.ma.masked_array(drifter_distribution, drifter_distribution==0)   

#Distribution of initial positions (fig. 1b)
drifter_data.compute_initial_symbols(d_deg = d_deg)
drifter_distribution0 = drifter_data.initial_distribution

#drifter lifetimes  (fig. 1c)
lifetimes = np.array([t[-1]-t[0] for t in drifter_data.drifter_time]) / 86400
lifetimes = lifetimes //(30.5)

#Plot figure
f = plt.figure(constrained_layout=True, figsize = (9,3))
gs = f.add_gridspec(1, 3)

ax1 = f.add_subplot(gs[0, 0])
ax1.set_title('(a) total data', size=12)
norm = colors.Normalize(0,20000)

drifter_data.plot_discretized_distribution_geo(ax1, drifter_distribution, cmap='OrRd', norm = norm)

norm = colors.Normalize(0,20)
ax2 = f.add_subplot(gs[0, 1])
ax2.set_title('(b) drifter release', size=12)
drifter_data.plot_discretized_distribution_geo(ax2, drifter_distribution0, cmap='OrRd', norm=norm)

norm = colors.Normalize(0,24)
ax3 = f.add_subplot(gs[0, 2])
ax3.set_title('(c) drifter lifetime (months)', size=12)
drifter_data.scatter_position_with_labels_geo(ax3, lifetimes, norm=norm, size=3, t=0, 
                                              cmap = 'cividis', alpha=1)

plt.savefig('figures/na_drifter_dataset_info', dpi=300)