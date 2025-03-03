import matplotlib.pyplot as plt

from kmeans import KMeans
from kneed import KneeLocator

from sklearn.datasets import make_blobs

NSAMPLES = 10**6
NFEATURES = 50
CENTERS = 20

X, y_true = make_blobs(
    n_samples=NSAMPLES,
    n_features=NFEATURES,
    centers=CENTERS,
    random_state=0,
)

sse = []
k_range = range(2, int(CENTERS * 1.5))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)


plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker="o")
plt.title("Método do cóbado")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia ou SSE")


kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")

print(f"The elbow is at {kl.elbow}")

plt.show()
