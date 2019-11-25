import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

'''
AgglomerativeClustering(n_clusters=2, affinity=’euclidean’, memory=None, 
                        connectivity=None, compute_full_tree=’auto’, 
                        linkage=’ward’,pooling_func=’deprecated’)
'''

dataset = load_iris()
data = dataset.data

agglomerative_cluster_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
y_pred = agglomerative_cluster_model.fit_predict(data)

plt.scatter(data[y_pred == 0, 0], data[y_pred == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(data[y_pred == 1, 0], data[y_pred == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(data[y_pred == 2, 0], data[y_pred == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
#plt.scatter(data[y_pred == 3, 0], data[y_pred == 3, 1], s = 50, c = 'magenta', label = 'Cluster 4')


plt.title('original data')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show()

print("-------------------------------------------")

dendrogram = sch.dendrogram(sch.linkage(data[0:30,:], method = 'average'))
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show()
