import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.decomposition import PCA as skelearnPca

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


csv_file = "FaceBook-dataset.csv"
df = pd.read_csv(csv_file, skipinitialspace=True)
df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)
df.drop(['status_id', 'status_published'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

x = df
y = df['status_type']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x['status_type'] = le.fit_transform(x['status_type'])

y = le.transform(y)
cols = x.columns
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

x = ms.fit_transform(x)
x = pd.DataFrame(x, columns=[cols])

from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=4,random_state=0)

kmeans.fit(x)

y_pred = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == y_pred)
centroids = kmeans.cluster_centers_

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
pca = skelearnPca(n_components=2)
bes_data = pca.fit_transform(x)
plt.scatter(bes_data[y_pred == 0, 0], bes_data[y_pred == 0, 1], s=100, c='yellow', label='Cluster 1')
plt.scatter(bes_data[y_pred == 1, 0], bes_data[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(bes_data[y_pred == 2, 0], bes_data[y_pred == 2, 1], s=100, c='purple', label='Cluster 3')
plt.scatter(bes_data[y_pred == 3, 0], bes_data[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='d', label='Centroids')

plt.xlabel('best 1')
plt.ylabel('best 2')
plt.legend()
plt.show()