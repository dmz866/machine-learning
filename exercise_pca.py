import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data # UPPERCASE for dataframes
y = iris.target

X_centrado = X - np.mean(X, axis = 0) #Standarize data

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_centrado)

especies = ['setosa', 'versicolor', 'virginica']

#draw
plt.figure(figsize=(8,6))

for i in range(0, 3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=especies[i])

plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.legend()
plt.title('PCA del conjunto Iris')
plt.show()



