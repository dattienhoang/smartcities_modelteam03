# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:42:56 2017

@author: Dat Tien Hoang
"""

import numpy as np
import pandas as pd

def rdf(coord_sta, coord_inp, rng=0.5, label_sta=None, plot_bin = 0.170):
    # assumptions:
    # coord_sta is a list as [lon, lat]
    # coord_inp is a two column numpy array with labels [lon, lat]
    # rng is a distance in miles, represents the maximum distance you are looking at
    #     set to None type if 
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
    # convert the coord_sta to numpy array
    coord_sta = np.asarray(coord_sta)
    # compute all the distances...then filter for those that are less than specified range 
    res = np.zeros(len(coords_inp))
    for i in range(len(res)):
        res[i] = haversine(coord_sta[0], coord_sta[1], coord_in[i,0], coord_inp[i,1])
    if rng != None:
        w = np.where(res < rng)
        # make sure that res is a numpy array
        res = res[w]
    # now make a plot that shows frequency histogram
    # in NYC, average block is 264 x 900 ft
    # 1 mile = 5280 ft, hence the default
    
    