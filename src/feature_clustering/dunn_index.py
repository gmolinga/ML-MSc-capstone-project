from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances

evaluation_indices = {
    'Silhouette Coefficient': silhouette_score,
    'Calinski-Harabasz Index': calinski_harabasz_score,
    'Davies-Bouldin Index': davies_bouldin_score
}

def dunn_index(data, labels):
    distances = pairwise_distances(data)
    intra_cluster_distances = []
    for label in set(labels):
        cluster_points = data[labels == label]
        cluster_distances = distances[labels == label][:, labels == label]
        intra_cluster_distances.append(cluster_distances.max())
    min_inter_cluster_distance = distances[labels != labels[:, None]].min()
    dunn_index = min_inter_cluster_distance / max(intra_cluster_distances)
    return dunn_index