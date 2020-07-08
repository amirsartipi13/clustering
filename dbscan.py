
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA as skelearnPca
from Sql import SqlManager
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_info(columns):
    sql_manager = SqlManager("information.sqlite")
    try:
        sql_manager.crs.execute("delete from encoding_guide")
        sql_manager.conn.commit()
    except:
        pass
    query = generate_query(columns)
    data = sql_manager.crs.execute(query).fetchall()
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    return data

def generate_query(columns):
    query = 'select {}'.format(columns[0])
    for i in range(1, len(columns)):
        query = query + ',' + str(columns[i])
    query = query + ' from information'
    return query


def find_best_col(data):
    pca = skelearnPca(n_components=2)
    data = pca.fit_transform(data)
    return data

def dbscan(eps, address):
    columns = ['status_type','num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']


    # columns = ['num_comments', 'num_likes']
    data = get_info(columns)
    best_data = find_best_col(data)
    ##############################################################################
    # Compute similarities
    D = distance.squareform(distance.pdist(data))
    S = 1 - (D / np.max(D))


    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=10).fit(best_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    dbs = davies_bouldin_score(best_data, labels)
    print('davies_bouldin_score for dbscan is : '+str(dbs))

    X = []
    Y = []

    for xy in best_data:
        X.append(xy[0])
        Y.append(xy[1])

    print(len(X))
    plt.scatter(X, Y, c=labels)

    plt.xlabel('best 1')
    plt.ylabel('best 2')
    plt.legend()
    plt.savefig('outs\\dbscan\\' + 'db' + '.png')


    plt.show()


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, labels))


    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #     xy = best_data[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #     xy = best_data[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters_)

if __name__ == "__main__":
    dbscan(eps=0.3, address='outs\\dbscan\\')