# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:42:56 2017

@author: Dat Tien Hoang
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rdf(coord_sta, coord_inp, rng=0.5, wt=1, label_sta=None, plot_bin = 0.170/3):
    # assumptions:
    # coord_sta is a list as [lon, lat]
    # coord_inp is a two column numpy array with labels [lon, lat]
    # rng is a distance in miles, represents the maximum distance you are looking at
    #     set to None type if 
    # label_sta is the label for the subway station of coord_sta
    from math import radians, cos, sin, asin, sqrt
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        km = 6367. * c
        # convert from km to miles
        dist = km/0.621371
        return dist
    if label_sta == None:
        label_sta = '?? subway station ??'
    # convert the coord_sta to numpy array
    coord_sta = np.asarray(coord_sta)
    # compute all the distances...then filter for those that are less than specified range 
    res = np.zeros(len(coord_inp))
    for i in range(len(res)):
        res[i] = haversine(coord_sta[0], coord_sta[1], coord_inp[i,0], coord_inp[i,1])
    
    if rng != None:
        w = np.where(res < rng)
        # make sure that res is a numpy array
        res = res[w]
    else:
        w = range(len(res))
    # now make a plot that shows frequency histogram
    # in NYC, average block is 264 x 900 ft
    # 1 mile = 5280 ft, hence the default
    # also have a subplot that just shows the spatial distribution 

    # remember to apply the weight to this station!
    plot = False
    if plot == True:
        plt.figure()
        plt.hist(res, bins=np.arange(0, max(res) + plot_bin, plot_bin))
        plt.xlabel('miles from station')
        plt.ylabel('crime occurence')
        plt.title('Radial Distribution Function of Crime at Station: ' + label_sta)
        
        plt_typ = 'scatter'
        plt.figure()
        if plt_typ == 'scatter':
            plt.scatter(coord_inp[w,0], coord_inp[w,1],color='g')
            plt.scatter(coord_sta[0], coord_sta[1],color='r', marker='o')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.title('Spatial Distribution of Crime at Station: ' + label_sta)
            axes = plt.gca()
            xrng = axes.get_xlim()
            yrng = axes.get_ylim()
            plt.show()
        if plt_typ == 'heatmap':
            import seaborn as sns
            g = sns.jointplot(coord_inp[w,0], coord_inp[w,1], kind="kde", xlim=xrng, ylim=yrng)
    
    hist, bin_edges = np.histogram(res, bins=np.arange(0, max(res) + plot_bin, plot_bin))
    hist = np.asarray(hist)
    bin_edges = np.asarray(bin_edges)
    norm = True
    if norm == True:
        hist = hist / sum(hist)
    return hist, bin_edges
    

#dummy_sta = [40.756266207, -73.990501248]
#dummy_inp = np.asarray([
#        [40.828754623, -73.866593516],
#        [40.809859893, -73.937644103],
#        [40.719711494, -73.9894242],
#        [40.694514975, -73.849134227],
#        [40.649370541, -73.960872294]
#        ])
#
#rdf(dummy_sta, dummy_inp, rng=None, wt=1)
#dummy_sta = [
#        [-73.998091, 40.660397], # 
#        [-73.87255, 40.689941], #
#        [	-73.981929, 40.768247],
#        [	-73.944216, 40.824783],
#        [	-73.810708, 40.70546]
#        #[-73.839718, 40.682174],
#        #[-73.945359, 40.815731],
#        #[-73.792691, 40.707255]
#        ]
#label = [
#        'BMT, 4 Avenue Line, 25th St',
#        'BMT, Broadway Jamaica Line, Cypress Hills',
#        'BMT IRT, Broadway-7th Ave Line, 59th St-Columbus Circle',
#        'IND, 8 Avenue, 145th St',
#        'IND, Queens Boulevard, Sutphin Blvd'
#        #'106th precinct, nearest Cypress Hills',
#        #'32nd precinct, nearest 145th St',
#        #'103rd precinct, nearest Sutphin Blvd'
#        ]

df = pd.read_csv('C:\Users\Dat Tien Hoang\Downloads\NYPD_Complaint_Data_Current_YTD.csv')
df = df[['Longitude', 'Latitude']].as_matrix()

dummy_sta = pd.read_csv('C:\Users\Dat Tien Hoang\Downloads\NYC_Transit_Subway_Entrance_And_Exit_Data.csv')
dummy_sta = dummy_sta[['Station Longitude', 'Station Latitude']].as_matrix()


counter = 0.
for i in range(len(dummy_sta)):
    r1, r2 = rdf(dummy_sta[i], df, rng=.5, wt=1, label_sta='')#label[i])
    if max(r1) != 1.0:
        if i == 0:
            hist = r1
        else:
            hist = hist + r1
        counter += 1.
# now take the average
hist /= counter

# do some plot
plt.figure()
plt.plot(r2[1:len(r2)], r1)
plt.xlabel('miles from station')
plt.ylabel('crime occurence')
plt.title('Radial Distribution Function of Crime at over stations')