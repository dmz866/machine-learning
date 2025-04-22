import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(42)
datos = np.random.rand(10,2)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(datos)

plt.plot(datos, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()