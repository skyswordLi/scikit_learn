import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


X = [[9670250, 1392358258],
     [2980000, 1247923065],
     [9629091, 317408015],
     [8514877, 201032714],
     [377873, 127270000],
     [7692024, 23540517],
     [9984670, 34591000],
     [17075400, 143551289],
     [513115, 67041000],
     [181035, 14805358],
     [99600, 50400000],
     [120538, 24052231]]

X = np.array(X)
a = X[:, : 1] / 17075400 * 10000
b = X[:, 1:] / 1392358258 * 10000
X = np.concatenate((a, b), axis=1)
cls = DBSCAN(eps=2000, min_samples=1).fit(X)
n_clusters = len(set(cls.labels_))
markers = ['^', 'x', 'o', '*', '+']

for i in range(n_clusters):
    my_members = cls.labels_ == i
    plt.scatter(
        X[my_members, 0], X[my_members, 1], s=60,
        marker=markers[i], c='b', alpha=0.5)
plt.title('dbscan')
plt.show()
