import numpy as np
import random
import tqdm


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 10,
        existed_initialization: bool = False,
        random_state: int = 0,
    ) -> None:
        """
        Initialize the KMeans object

        Args:
            - n_clusters (int): number of clusters
            - max_iter (int): maximum number of iterations
            - existed_initialization (bool): if True, centroids are initialized randomly
            - random_state (int): random seed

        Returns:
            - None
        """
        self.k = n_clusters
        self.max_iter = max_iter
        self.existed_initialization = existed_initialization

        # fijamos la semilla para obtener resultados reproducibles
        random.seed(random_state)
        np.random.seed(random_state)

        # atributos que se definen en el fit
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Aplica el algoritmo de entrenamiento de K-Means

        Args:
            - X (np.ndarray): puntos de datos
                Las filas son las instancias y
                las columnas son las características

        Returns:
            - None
        """
        n_inter = 0
        centroids = self._initialize_centroids(X)

        # Para el criterio de parada
        oldCentroids = np.zeros_like(centroids)

        # Bucle principal
        with tqdm.tqdm(total=self.max_iter, desc=f"{self.k}means") as pbar:
            while n_inter < self.max_iter and not np.array_equal(
                centroids, oldCentroids
            ):

                # Guardamos los centroides antiguos
                oldCentroids = centroids

                labels = self._get_labels(X, centroids)
                centroids = self._update_centroids(X, labels, centroids)
                n_inter += 1
                pbar.update(1)

        # Guardamos los resultados
        self.cluster_centers_ = centroids
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        self.labels_ = labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica el algoritmo de predicción de K-Means

        Args:
            - X (np.ndarray): puntos de datos
                Las filas son las instancias y
                las columnas son las características

        Returns:
            - np.ndarray: etiquetas de los puntos
        """
        labels = self._get_labels(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Entrena con los datos y después
        predice las etiquetas de los mismos

        Args:
            - X (np.ndarray): puntos de datos
                Las filas son las instancias y
                las columnas son las características

        Returns:
            - np.ndarray: etiquetas de los puntos
        """
        self.fit(X)
        return self.predict(X)

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Inicializa los centroides de los clusters

        Para esta función es importante el atributo
        self.existed_initialization. Si es True, se
        escogen aleatoriamente k puntos de X como centroides.
        Si es false, se usa una distribución uniforme
        para generar los centroides.

        Nos aseguramos de que no haya centroides repetidos

        Args:
            - X (np.ndarray): puntos de datos

        Returns:
            - np.ndarray: centroides de los clusters
        """
        if self.existed_initialization:
            # escogemos aleaotoriamente k puntos de X como centroides
            idx = np.random.choice(X.shape[0], self.k, replace=False)
            centroids = X[idx]
        else:
            # obtenemos los mínimos y máximos valores para cada atributo
            lower_bound = np.min(X, axis=0)
            upper_bound = np.max(X, axis=0)

            # No queremos centroides repetidos
            centroids = set()

            # Generamos k centroides
            while len(centroids) < self.k:
                # Generamos un centroide aleatorio
                centroid = np.random.uniform(
                    low=lower_bound, high=upper_bound, size=(1, X.shape[1])
                )
                centroids.add(tuple(centroid[0]))

            centroids = np.array(list(centroids))

        return centroids

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Actualiza los centroides de los clusters

        Si no hay puntos asignados a un cluster, el centroide
        se mantiene igual.

        Args:
            - X (np.ndarray): puntos de datos
            - labels (np.ndarray): etiquetas de los puntos
            - centroids (np.ndarray): centroides antiguos de los clusters

        Returns:
            - np.ndarray: nuevos centroides de los clusters
        """
        new_centroids = np.array(
            [
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(len(centroids))
            ]
        )

        return new_centroids

    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de cada punto a cada centroide

        Usa la distancia euclidiana.

        Args:
            - X (np.ndarray): puntos de datos
            - centroids (np.ndarray): centroides de los clusters

        Returns:
            - np.ndarray: matriz de distancias
                Las filas son los puntos y
                las columnas son los centroides
        """
        distances = np.array(
            [np.linalg.norm(X - centroid, axis=1) for centroid in centroids]
        ).T
        return distances

    def _get_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calcula el centroide más cercano de cada punto

        En caso de empate, se asigna el cluster con menos puntos.
        Esto es horrible para la eficiencia, lo mejor es hacer
        el mínimo con argmin y ya está.
        Eso es mucho más paralelizable.

        Args:
            - X (np.ndarray): puntos de datos
            - centroids (np.ndarray): centroides de los clusters

        Returns:
            - np.ndarray: etiquetas de los puntos
        """
        distances = self._compute_distance(X, centroids)
        labels = np.argmin(distances, axis=1)

        # comprobamos que no haya casos con varias distancias mínimas
        for i in range(X.shape[0]):
            label = labels[i]
            current_distance = distances[i][label]

            candidates = np.where(distances[i] == current_distance)[0]

            if len(candidates) > 1:
                counts = np.array(
                    [np.sum(labels == candidate) for candidate in candidates]
                )
                best_candidate = candidates[np.argmin(counts)]
                labels[i] = best_candidate

        return labels

    def _compute_inertia(
        self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        """
        Calcula la suma de las distancias de cada punto al centroide correspondiente

        Args:
            - X (np.ndarray): puntos de datos
            - labels (np.ndarray): etiquetas de los puntos
            - centroids (np.ndarray): centroides de los clusters

        Returns:
            - float: suma de las distancias de los puntos al centroide correspondiente
        """
        return np.sum(np.linalg.norm(X - centroids[labels], axis=1))


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
