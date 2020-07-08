# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Sql import SqlManager
from sklearn.decomposition import PCA as skelearnPca
from sklearn.metrics import davies_bouldin_score
import  pandas as pd

def get_info(columns):
    sql_manager = SqlManager("information.sqlite")
    try:
        sql_manager.crs.execute("delete from encoding_guide")
        sql_manager.conn.commit()
    except:
        pass
    query = generate_query(columns)
    data = sql_manager.crs.execute(query).fetchall()
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
    columns = ['status_type', 'num_comments', 'num_reactions',
            'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys']
    # columns = ['num_comments', 'num_likes']

    query = generate_query(columns)
    column_value = sql_manager.crs.execute(query).fetchall()
    wcss = []

    for i in range(1, cluster_count_test + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=50, max_iter=max_iter, random_state=0)
        kmeans = kmeans.fit(column_value)
        wcss.append(kmeans.inertia_)
    ellbow_plot(wcss, cluster_count_test, max_iter)


def find_best_col(data):
    pca = skelearnPca(n_components=2)
    data = pca.fit_transform(data)
    print(data)
    return data

def k_means(cluster_count, max_iter):
    columns = ['status_type','num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']

    # columns = ['num_comments', 'num_likes']
    data = get_info(columns)
    bes_data = find_best_col(data)
    data = pd.DataFrame(data, columns=columns)

    kmeans = KMeans(n_clusters=cluster_count, init='k-means++', n_init=50, max_iter=max_iter, random_state=0)
    kmeans = kmeans.fit(bes_data)

    y_pred = kmeans.predict(bes_data)  # predicted labels
    centroids = kmeans.cluster_centers_

    dbs = davies_bouldin_score(bes_data, y_pred)

    X = []
    Y = []

    for xy in bes_data:
        X.append(xy[0])
        Y.append(xy[1])

    print(len(X))
    plt.scatter(X, Y, c=y_pred)
    x_centers = [x[0] for x in centroids]
    y_centers = [y[1] for y in centroids]
    plt.scatter(x_centers, y_centers, c="r", marker="+", s=200)
    plt.xlabel('best 1')
    plt.ylabel('best 2')

    plt.legend()
    plt.savefig('outs\\k-means\\'+'max iter '+str(max_iter)+' and cluster count '+str(cluster_count)+'.png')
    plt.show()
    print('davies_bouldin_score for k-means is : '+str(dbs))

if __name__ == "__main__":

    #ellbow(cluster_count_test=15, max_iter=10)
    k_means(cluster_count=4, max_iter=30)
