import numpy as np
import random


class KMeans:
    def __init__(
        self, n_clusters, max_iter=10, existed_initialization=False, random_state=0
    ):
        self.k = n_clusters
        self.max_iter = max_iter
        self.existed_initialization = existed_initialization
        random.seed = random_state
        np.random.seed(random_state)

        # atrubutos que se definen en el fit
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Aplica el algoritmo de entrenamiento de K-Means
        :param X: puntos de datos
        """
        n_inter = 0
        centroids = self._initialize_centroids(X)
        while n_inter < self.max_iter:
            labels = self._get_labels(X, centroids)
            centroids = self._update_centroids(X, labels, centroids)
            n_inter += 1

        self.cluster_centers_ = centroids
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        self.labels_ = labels

    def predict(self, X):
        """
        Aplica el algoritmo de predicción de K-Means
        :param X: puntos de datos
        :return: etiquetas de los puntos
        """
        labels = self._get_labels(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        """
        Aplica el algoritmo de entrenamiento y predicción de K-Means
        :param X: puntos de datos
        :return: etiquetas de los puntos
        """
        self.fit(X)
        return self.predict(X)

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Inicializa los centroides de los clusters
        :param X: puntos de datos
        :return: centroides de los clusters
        """
        if self.existed_initialization:
            # escogemos aleaotoriamente k puntos de X como centroides
            idx = np.random.choice(X.shape[0], self.k)
            centroids = X[idx]
        else:
            # obtenemos los mínimos y máximos valores para cada atributo
            lower_bound = np.min(X, axis=0)
            upper_bound = np.max(X, axis=0)
            # generamos los centroides aleatorios
            centroids = np.random.uniform(
                low=lower_bound, high=upper_bound, size=(self.k, X.shape[1])
            )
        return centroids

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Actualiza los centroides de los clusters
        :param X: puntos de datos
        :param labels: etiquetas de los puntos
        :param centroids: centroides de los clusters
        :return: centroides de los clusters
        """
        new_centroids = np.zeros(shape=(self.k, X.shape[1]))
        for i in range(self.k):
            # obtenemos los puntos de un cluster
            points = X[labels == i]
            if points.shape[0] > 0:
                new_centroids[i] = np.mean(points, axis=0, dtype=np.float64)
            else:  # caso en que no hay puntos de un cluster
                new_centroids[i] = centroids[i]
        return new_centroids

    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            # calcula la distancia euclidiana entre los puntos y el centroide i
            # la distancia es la norma euclideana de la diferencia entre los puntos y el centroide
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        return distances

    def _get_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula el centroide más cercano de cada punto
        :param X: puntos de datos
        :param centroids: centroides de los clusters
        :return: etiquetas de los puntos
        """
        distances = self._compute_distance(X, centroids)
        labels = np.argmin(distances, axis=1)

        # comprobamos que no haya casos con varias distancias mínimas
        for i in range(X.shape[0]):
            label = labels[i]
            current_distance = distances[i][label]
            for j in range(label, self.k):
                if distances[i, j] == current_distance:
                    # comprobamos que cluster lleva menos elementos
                    if np.sum(labels == j) < np.sum(labels == label):
                        label = j

            labels[i] = label

        return labels

    def _compute_inertia(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        """
        Calcula la suma de las distancias de cada punto al centroide correspondiente
        :param X: puntos de datos
        :param labels: etiquetas de los puntos
        :param centroids: centroides de los clusters
        :return: suma de las distancias de los puntos al centroide correspondiente
        """
        inertia = 0
        for i in range(self.k):
            points = X[labels == i]
            inertia += np.sum(np.linalg.norm(points - centroids[i], axis=1))
        return inertia


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    """
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    kmeans = KMeans(k=2, max_iter=10, existed_initialization=False, random_seed=0)
    kmeans.fit(X)
    """
    puntos = [(2, 5), (8, 4), (7, 5), (6, 4), (4, 9)]
    clusters = [(2, 10), (5, 8), (1, 2)]
    puntos = np.array(puntos)
    clusters = np.array(clusters)
    kmeans = KMeans(
        n_clusters=3, max_iter=10, existed_initialization=False, random_state=0
    )
    print(kmeans._get_labels(puntos, clusters))
