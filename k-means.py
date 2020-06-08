# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Sql import SqlManager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skelearnPca
from sphinx.addnodes import not_smartquotable

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


def ellbow_plot(wcss, cluster_count_test, max_iter):
    # Plot wccs values wrt number of clusters
    plt.plot(np.arange(1, cluster_count_test+1), wcss, '-o')
    plt.title('ELbow of KMeans')
    plt.xlabel('Number of Clusters, k')
    plt.ylabel('inertia')
    plt.xticks(np.arange(1, cluster_count_test+1))
    plt.savefig('outs\\plots\\'+'ellbow_'+str(max_iter))
    plt.close()

def generate_query(columns):
    query = 'select {}'.format(columns[0])
    for i in range(1, len(columns)):
        query = query + ',' + str(columns[i])
    query = query + ' from information'
    return query

def ellbow(cluster_count_test, max_iter):
    sql_manager = SqlManager("information.sqlite")
    try:
        sql_manager.crs.execute("delete from encoding_guide")
        sql_manager.conn.commit()
    except:
        pass
    columns = ['num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']
    # columns = ['num_comments', 'num_likes']

    query = generate_query(columns)
    column_value = sql_manager.crs.execute(query).fetchall()

    # scaler = StandardScaler()
    # column_value = scaler.fit_transform(column_value)

    wcss = []
    for i in range(1, cluster_count_test + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=50, max_iter=max_iter, random_state=0)
        kmeans = kmeans.fit(column_value)
        wcss.append(kmeans.inertia_)
    ellbow_plot(wcss, cluster_count_test, max_iter)


def find_best_col(data):
    pca = skelearnPca(n_components=2)
    data = pca.fit_transform(data)
    return data

def k_means(cluster_count, max_iter):
    columns = ['num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']

    # columns = ['num_comments', 'num_likes']
    data = get_info(columns)
    best_data = find_best_col(data)

    print(best_data)
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++', n_init=50, max_iter=max_iter, random_state=0)
    kmeans = kmeans.fit(best_data)

    y_pred = kmeans.predict(best_data)  # predicted labels
    centroids = kmeans.cluster_centers_

    # Plot scatter of datapoints with their clusters
    plt.scatter(best_data[y_pred == 0, 0], best_data[y_pred == 0, 1], s=100, c='yellow', label='Cluster 1')
    plt.scatter(best_data[y_pred == 1, 0], best_data[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(best_data[y_pred == 2, 0], best_data[y_pred == 2, 1], s=100, c='purple', label='Cluster 3')
    plt.scatter(best_data[y_pred == 3, 0], best_data[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(best_data[y_pred == 4, 0], best_data[y_pred == 4, 1], s=100, c='green', label='Cluster 5')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='d', label='Centroids')

    plt.xlabel('best 1')
    plt.ylabel('best 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # ellbow(cluster_count_test=15, max_iter=1)
    k_means(cluster_count=5, max_iter=10)
