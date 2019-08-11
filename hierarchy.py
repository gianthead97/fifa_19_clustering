#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from  sklearn.metrics import silhouette_score
import numpy as np




df = pd.read_csv('data.csv', index_col=0)
df_scaled = pd.read_csv('preprocessed_data.csv', index_col=0)




colors = ['deeppink', 'darkgreen', 'brown', 'cyan',
         'grey', 'red', 'blue', 'navy',
         'black', 'orange', 'orchid']




# for link in ['complete', 'average', 'single']:
#     for n in range(3, 10):
#         est_agg = AgglomerativeClustering(n_clusters=n, linkage = link, affinity='euclidean')
#         est_agg.fit(df_scaled)
#         df['Labels'] = est_agg.labels_
#         print(link)
#         print(n)
#         print(silhouette_score(df_scaled, est_agg.labels_))




position_to_num = {
    'GK': 0.0,
    'CB': 1.0,
    'LCB': 1.2,
    'RCB': 1.6,
    'LB': 2.7,
    'RB': 3.2,
    'LWB': 4.5,
    'RWB': 4.6,
    'CM': 6,
    'LCM': 6.2,
    'RCM': 6.4,
    'CDM': 5,
    'LDM': 5.1,
    'RDM': 5.3,
    'LM': 6.5,
    'RM': 6.7,
    'RAM': 7.3,
    'CAM': 7,
    'LAM': 7.1,
    'LW': 8.2,
    'RW': 8.4,
    'CF': 9.1,
    'LF': 9.2,
    'RF': 9.4,
    'LS': 9.5,
    'RS': 9.7,
    'ST': 10
}
df['Position'].replace(position_to_num, inplace=True)




from sklearn.decomposition import PCA




pca = PCA(n_components=3)
pca_df = pca.fit_transform(df_scaled)
pca.explained_variance_
df_pca = pd.DataFrame(pca_df, columns=['pca1', 'pca2', 'pca3'])




est_agg = AgglomerativeClustering(n_clusters=11, linkage = 'single', affinity='euclidean')
est_agg.fit(df_scaled)
df['Labels'] = est_agg.labels_
        




from mpl_toolkits.mplot3d import Axes3D





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(11):
    cluster = df.sample(2000)[df['Labels'] == i]
    ax.scatter(cluster['Overall'], cluster['Age'], cluster['Stamina'],
                   c=colors[i])

ax.set_xlabel('Overall')
ax.set_ylabel('Age')
ax.set_zlabel('Stamina')
plt.show()
#plt.savefig('../agglomerative_11_pca_wh')


