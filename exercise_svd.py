import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

X_centrado = X - np.mean(X, axis=0)
U, sigma, VT = np.linalg.svd(X_centrado)

k = 2
X_transformado = U[:, :k] * sigma[:k]

especies = ['setosa', 'versicolor', 'virginica']

#draw
plt.figure(figsize=(8,6))

for i in range(3):
    plt.scatter(X_transformado[iris.target == i, 0], X_transformado[iris.target == i, 1], label=especies[i])

plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.legend()
plt.title('Dataset iris transformado por SVD')
plt.show()


