# Code for "Detecting flow features in scarce trajectory data using networks derived from symbolic itineraries: an application to surface drifters in the North Atlantic"
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

Questions to: d.wichmann@uu.nl

# Code description:
particle_and_network_classes.py: Main script to handle trajectory data and for network analysis and

## Preparation
Set up python environment specified in 'env_drifter_network.yml'

## North Atlantic drifters
### Preparation of drifter data
1. Download the data (.dat) from https://www.aoml.noaa.gov/phod/gdp/interpolated/data/all.php
2. Execute 'constrain_drifterdata_to_northatlantic.py', possibly adjusting the name of the .dat files (if updated). This saves a file 'drifterdata_north_atlantic.npz' containing only thos drifters that start in the North Atlantic. Note that this script takes a whil, possible more than an hour. It is not efficiently implemented at the moment.

### Plot figure 1 (drifter info)
Execute 'drifter_data_info.py'

### Clustering (figures 6,7, C1)
Execute 'clustering_north_atlantic_drifters.py' with the option plot_365days =True (figs 6,7 ) and False (fig. C1)

## Double-gyre flow
