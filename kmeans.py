from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv', index_col = 0)
df_preprocessed = pd.read_csv('preprocessed_data.csv', index_col = 0)

def fix_value(x: str) -> str:
    if x.find('.') == -1:
        return x
    else:
        return x[:len(x) - 1].replace('.', '')


df['Value'] = df['Value'].replace({
    '€': '',
    'M': '000000',
    'K': '000'
}, regex=True).map(fix_value).convert_objects(convert_numeric=True)

position_to_num = {
    'GK': 0.0,
    'CB': 1.0,
    'LCB': 1.2,
    'RCB': 1.6,
    'LB': 2.7,
    'RB': 3.2,
    'RWB': 4.5,
    'LWB': 4.6,
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

est_kmeans = KMeans(n_clusters=4).fit(df_preprocessed.drop('Value', axis = 1))
df['Labels'] = est_kmeans.labels_

colors = ['deeppink', 'darkgreen', 'gold', 'cyan']
fig = plt.figure(figsize=(20, 12))

for j in range(4):
    cluster = df.sample(300)[df['Labels'] == j]
    plt.scatter(cluster['Position'], cluster['Overall'], color = colors[j], label ='Cluster %d'%j, s = 190, alpha=1)

plt.axvspan(-0.1, 0.9, facecolor='green', alpha = 0.3)
plt.axvspan(0.9, 4.6, facecolor='green', alpha = 0.4)
plt.axvspan(4.6, 7.1, facecolor='green', alpha = 0.5)
plt.axvspan(7.1, 10.1, facecolor='green', alpha = 0.6)
plt.tick_params(axis='y', which='major')


plt.title('Klasterovanje na osnovu 42 atributa u 4 klastera')
plt.legend()
plt.xlabel("Pozicija na terenu")
plt.ylabel("Overall")
plt.tight_layout()
plt.show()

