import matplotlib.pyplot as plt
import numpy as np

from kmeans import KMeans

from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

X.shape

plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.title("Dataset sintético de mostra")
plt.show()

sse = []
k_range = range(2, 11)

for k in k_range:
    kmeans2 = KMeans(n_clusters=k, random_state=0)
    kmeans2.fit(X)
    sse.append(kmeans2.inertia_)


plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker="o")
plt.title("Método do cóbado")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia ou SSE")
plt.show()

from kneed import KneeLocator

kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")

kl.elbow

from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in k_range:
    kmeans3 = KMeans(n_clusters=k)
    kmeans3.fit(X)
    score = silhouette_score(X, kmeans3.labels_)
    silhouette_coefficients.append(score)


plt.figure(figsize=(8, 6))
plt.plot(k_range, silhouette_coefficients, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette")
plt.show()


# Crea un modelo de K-Means con 4 clusters
kmeans = KMeans(n_clusters=4)

# Axustamos o modelo
kmeans.fit(X)

# Produce a saída etiquetada
y_kmeans = kmeans.predict(X)

# Con fit_predict() podemos facer todo xunto
# y_kmeans = kmeans.fit_predict(X)
#
# Podemos acceder ás etiquetas sen chamar a predict(), mediante kmeans.labels_
# kmeans.labels_


for i in range(5):
    print(f"Punto {i}: {X[i]} no cluster {y_kmeans[i]}")

np.unique(y_kmeans, return_counts=True)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap="viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="x")

labels = KMeans(4, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="managua")
plt.show()
