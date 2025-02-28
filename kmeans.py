import numpy as np
import random


class KMeans:
    def __init__(self, k, max_iter=10, existed_initialization=False, random_seed=0):
        self.k = k
        self.max_iter = max_iter
        self.existed_initialization = existed_initialization
        random.seed = random_seed
        np.random.seed(random_seed)

    def fit(self, X):
        n_inter = 0
        centroids = self._initialize_centroids(X)
        while n_inter < self.max_iter:
            labels = self.get_labels(X, centroids)
            new_centroids = self._update_centroids(X, labels)

    def predict(self, X):
        pass

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.existed_initialization:
            idx = np.random.choice(X.shape[0], self.k)
            centroids = X[idx]
        else:
            # obtenemos los mínimos y máximos valores para el atributo 1 y 2
            lower_bound = np.min(X, axis=0)
            upper_bound = np.max(X, axis=0)
            # generamos los centroides aleatorios
            centroids = np.random.uniform(
                low=lower_bound, high=upper_bound, size=(self.k, X.shape[1])
            )
        return centroids

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros(self.k, X.shape[1])
        for i in range(self.k):
            pass

    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        print(distances)
        return distances

    def _get_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
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
    kmeans = KMeans(k=3, max_iter=10, existed_initialization=False, random_seed=0)
    print(kmeans._get_labels(puntos, clusters))
