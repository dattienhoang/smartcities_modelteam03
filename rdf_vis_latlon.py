# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:42:56 2017

@author: Dat Tien Hoang
"""

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
    print res
    if rng != None:
        w = np.where(res < rng)
        # make sure that res is a numpy array
        res = res[w]
    else:
        w = range(len(res))
    # now make a plot that shows frequency histogram
    # in NYC, average block is 264 x 900 ft
    # 1 mile = 5280 ft, hence the default
    # also have a subplot that just plain 
    print res
    # remember to apply the weight to this station!
    
    if plot_bin != False:
        #hist, bin_edges = np.histogram(res)#, bins=plot_bin)
        
        plt.figure()
        plt.hist(res, bins=np.arange(0, max(res) + plot_bin, plot_bin)) 
        
        plt.figure()
        plt.scatter(coord_inp[w,0], coord_inp[w,1],color='g')
        plt.scatter(coord_sta[0], coord_sta[1],color='k')
        plt.show()
        print w
        print coord_inp
    
    
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
#dummy_sta = [-73.998091, 40.660397]# BMT, 4 Avenue Line, 25th St
dummy_sta = [-73.87255, 40.689941] #BMT	Broadway Jamaica	Cypress Hills	
df = pd.read_csv('C:\Users\Dat Tien Hoang\Downloads\NYPD_Complaint_Data_Current_YTD.csv')
df = df[['Longitude', 'Latitude']].as_matrix()
rdf(dummy_sta, df, rng=.5, wt=1)