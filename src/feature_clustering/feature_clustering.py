import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from . import dunn_index

class FeatureCluster:

    def __init__(self, k="auto", init="k-means++", max_iter=100 ,standarize=True):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.standarize = standarize

    def fit(self, X, y):
        if self.standarize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        self.X = X
        self.y = y

        if self.k == "auto":
            tries = {}
            for try_k in range(2, int(math.sqrt(self.X.shape[1]))+1 ):
                kmeans = KMeans(n_clusters=try_k, init=self.init, max_iter=self.max_iter, n_init="auto")
                kmeans.fit(self.X.T)
                score = dunn_index.evaluation_indices["Silhouette Coefficient"](self.X.T, kmeans.labels_)
                tries[try_k] = {
                    "kmeans": kmeans,
                    "score": score,
                }
                print(f"KMeans with k '{try_k}' has a silhouette coefficient of '{score}'")
            best_iteration = sorted(tries.items(), key=lambda x: x[1]["score"], reverse=True)[0]
            self.k = best_iteration[0]
            self.kmeans = best_iteration[1]["kmeans"]
        self.kmeans = KMeans(n_clusters=self.k, init=self.init, max_iter=self.max_iter, n_init="auto")
        self.kmeans.fit(self.X.T)
        self.labels = self.kmeans.labels_

    def plot(self, plot_type="hist", **kwargs):
        if plot_type == "hist":
            pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
            sns.countplot(x=self.labels, palette= pal)
            plt.title(f"Distribución de los clusters para {self.k} clusters")
            plt.xlabel("Cluster")
            plt.ylabel("Frecuencia")
            plt.show()
        elif plot_type == "tsne":
            tsne = TSNE(n_components=2, **kwargs)
            embeddings = tsne.fit_transform(self.X.T)
            vis_data = pd.DataFrame(embeddings, index=self.X.T.index)
            vis_data["cluster"] = self.labels
            sns.scatterplot(
                x=0, y=1, hue="cluster", data=vis_data, palette=sns.color_palette("hls", 8)
            )
        elif plot_type == "umap":
            embeddings = umap.UMAP(n_neighbors=self.k, **kwargs).fit(self.X.T).embedding_
            vis_data = pd.DataFrame(embeddings, index=self.X.T.index)
            vis_data["cluster"] = self.labels
            sns.scatterplot(
                x=0, y=1, hue="cluster", data=vis_data, palette=sns.color_palette("hls", 8)
            )
        else:
            raise ValueError(f"Unknown value '{plot_type}' for parameter plot_type")

    def score(self):
        scores = {}

        for index_name, index_func in dunn_index.evaluation_indices.items():
            score = index_func(self.X.T, self.labels)  # Usa las etiquetas actuales
            scores[index_name] = score

        # Calcular los índices de Dunn y Hopkins.
        dunn = dunn_index.dunn_index(self.X.T, self.labels)  # Usa las etiquetas actuales
        scores['Dunn Index'] = dunn
        return scores

    def get_feature_clusters(self):
        unique_labels = set(self.labels)
        features_clusters = {
            label: self.X.T.index[self.labels == label].tolist()
            for label in unique_labels
        }
        return features_clusters
