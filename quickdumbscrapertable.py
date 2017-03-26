# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:27:45 2017

@author: Dat Tien Hoang
"""

from pandas.io.gbq import read_gbq

project = "spheric-crow-161317"
sample_query = "SELECT * FROM `smart_cities_data.NYC_MTA_stations` LIMIT 10"

df = read_gbq(query=sample_query, project_id=project, dialect='standard')