#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:09:43 2019

@author: nikjan
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv('data.csv', index_col=0)
df_scaled = pd.read_csv('preprocessed_data.csv', index_col=0)


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(50, 26))

dn = dendrogram(linkage(df_scaled.sample(300), 'average'), count_sort=True, no_labels=True)
plt.gcf()
plt.savefig('dendro1000')
plt.show()

