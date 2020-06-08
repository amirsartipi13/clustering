from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['num_comments', 'num_reactions',
           'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
           'num_sads', 'num_angrys']
csv_file = "FaceBook-dataset.csv"
df = pd.read_csv(csv_file, skipinitialspace=True)
x = df['num_comments']
y = df['num_likes']

plt.scatter(x,y)
plt.xlabel("num_comments")
plt.ylabel("num_likes")
plt.show()
df = df[columns]
# stscaler = StandardScaler().fit(df)
# df = stscaler.transform(df)
dbsc = DBSCAN(eps=0.3, min_samples = 15).fit(df)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True
for y in  core_samples:
    print(y)
plt.scatter(core_samples, labels)
plt.xlabel("num_comments")
plt.ylabel("num_likes")
plt.show()