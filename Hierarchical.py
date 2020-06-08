# import libraries
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Sql import SqlManager
from sklearn.decomposition import PCA as skelearnPca
import scipy.cluster.hierarchy as h
from sklearn.cluster import AgglomerativeClustering


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

def hierarchical():
    columns = ['num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']

    # columns = ['num_comments', 'num_likes']
    data = get_info(columns)
    best_data = find_best_col(data)

    # Finding the optimal count of clusters using Dendogram method
    dendogram = h.dendrogram(Z=h.linkage(best_data, method='ward'))
    print(dendogram)
    plt.title('Dendogram')
    plt.xlabel('Customers')
    plt.ylabel('Distance')
    plt.show()

    # Fitting Hierachial clustring using optimal number of clusters
    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_pred = hc.fit_predict(best_data)

    # Plot scatter of datapoints with their clusters
    plt.scatter(best_data[y_pred == 0, 0], best_data[y_pred == 0, 1], s=100, c='yellow', label='Cluster 1')
    plt.scatter(best_data[y_pred == 1, 0], best_data[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(best_data[y_pred == 2, 0], best_data[y_pred == 2, 1], s=100, c='purple', label='Cluster 3')
    plt.scatter(best_data[y_pred == 3, 0], best_data[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(best_data[y_pred == 4, 0], best_data[y_pred == 4, 1], s=100, c='green', label='Cluster 5')
    plt.title('Clusters (Hierachial))')
    plt.xlabel('Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    hierarchical()
