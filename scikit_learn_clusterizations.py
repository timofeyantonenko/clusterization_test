from sklearn.cluster import KMeans


def sl_k_means(num_of_clusters, input_data):
    clusterer = KMeans(n_clusters=num_of_clusters)
    cluster_labels = clusterer.fit_predict(input_data)
    return cluster_labels
