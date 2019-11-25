import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
NearestNeighbors(n_neighbors=5, radius=1.0, algorithm=’auto’,
                 leaf_size=30, metric=’minkowski’, p=2, metric_
                 params=None, n_jobs=None, **kwargs)
'''
data = np.array([
        [1,1],
        [1,2],
        [2,1],
        [2,3],
        [1,5],
        [6,8],
        [7,9],
        [6,9],
        [8,8],
        [8,10],
        [14,1],
        [14,2],
        [15,1],
        [15,3]])


nearest_neighbors_model = NearestNeighbors(n_neighbors=5,algorithm='auto',radius=1.0)
nearest_neighbors_model.fit(data)

#Calculating Details
print('NearestNeighborsModel kneighbors are : ' , nearest_neighbors_model.kneighbors(data))
print("-------------------------------------------------------")
print('NearestNeighborsModel radius kneighbors are : ' , nearest_neighbors_model.radius_neighbors(data))
