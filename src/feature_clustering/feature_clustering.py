import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
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
            raise ValueError(
                f"Unknown value '{plot_type}' for parameter plot_type. "
                "Valid values are: 'hist', 'tsne' and 'umap'."
            )

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

    def select_best_features(self, strategy="mutual-information"):
        if strategy=="mutual-information":
            best_predictors = {}
            for cluster_name, predictors in self.get_feature_clusters().items():
                print(f"Computing cluster {cluster_name}")
                cluster = self.X[predictors]
                mi = mutual_info_classif(cluster, self.y)
                mi_df = pd.DataFrame({
                    'variable': cluster.columns,
                    'mi': mi
                })
                mi_df = mi_df.sort_values(by='mi', ascending=False)
                top_50_percent = mi_df[:len(mi_df)//2]
                max_predictors = min(len(cluster.columns), 6)
                top_predictors = top_50_percent[:max_predictors]
                best_predictors[cluster_name] = top_predictors['variable'].tolist()
            return best_predictors
        elif strategy=="forward-feature-selection":

            clusters = self.get_feature_clusters()
            hyperparameters = {
                'n_estimators':10,
                'max_depth':5,
                'min_samples_split':20,
                'min_samples_leaf':10,
                'max_features':'sqrt',
                'bootstrap': True,
                'oob_score':False
            }

            variables_escogidas = []
            auc_actual = 0
            stopper_auc = 0
            smallest_cluser_size =  min([len(cluster) for cluster in clusters.values()])
            for j in range (0,smallest_cluser_size):
                if stopper_auc == 2:
                    break
                for i in clusters.keys():
                    active_cluster = clusters[i]
                    best_auc = 0                  
                    best_variable = []
                    if j >= 1 and len(variables_escogidas) >= 20:
                        break
                    for variable in active_cluster:
                        selected_features = variables_escogidas + [variable]
                        df_escogido = self.X[selected_features]
                        X_train, X_test, y_train, y_test = train_test_split(df_escogido, self.y, test_size=0.2, random_state=42)
                        scaler=StandardScaler()
                        scaler.fit(X_train)
                        under_sampler = RandomUnderSampler(random_state=42)
                        X_res, y_res = under_sampler.fit_resample(X_train, y_train)
                        model = RandomForestClassifier(**hyperparameters)
                        model.fit(X_res, y_res)
                        scaler2=StandardScaler()
                        scaler2.fit(X_test)
                        y_pred = model.predict(X_test)
                        auc = roc_auc_score(y_test, y_pred)
                        if auc >= best_auc:
                            best_variable = [variable]
                            best_auc = auc
                    
                    variables_escogidas+=best_variable
                    print(best_variable)
                    print("best_auc",best_auc,"auc_actual",auc_actual)
                    if j > 0 and best_auc < auc_actual:
                        if stopper_auc == 2:
                            break
                        if stopper_auc < 2:
                            stopper_auc = stopper_auc + 1
                    auc_actual = best_auc

            best_predictors = {id:[feature for feature in cluster if feature in variables_escogidas] for id, cluster in clusters.items()}
            return best_predictors
        else:
            raise ValueError(
                f"Unknown value '{strategy}' for parameter strategy. "
                "Valid values are: 'mutual-information' and 'forward-feature-selection'."
            )