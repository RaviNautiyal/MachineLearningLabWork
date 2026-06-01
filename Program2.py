import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris

# 1. KNN Implementation from scratch
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 2. Data Loading: IRIS Dataset
iris = load_iris()
X, y = iris.data, iris.target

# 3. Method 1: Square Root N
n_samples = X.shape[0]
k_sqrt_n = int(np.sqrt(n_samples))
if k_sqrt_n % 2 == 0: k_sqrt_n += 1
print(f"Total samples: {n_samples}")
print(f"K chosen by Square Root N rule: {k_sqrt_n}")