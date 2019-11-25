import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300,
tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, 
copy_x=True, n_jobs=None, algorithm=’auto’)
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


clusters_num = []
errors = []

#print('clusters labels : ',lables)
#print('error : ',error)
#print('predicted data \n ',predict)

for num in range(1,10):
    print("Clusters numbers = ",num)
    kmeans_model = KMeans(n_clusters=num)
    kmeans_model.fit(data)
    clusters_num.append(num)
    errors.append(kmeans_model.inertia_)
    print('error : ',kmeans_model.inertia_)

    print("--------------------------------------------------")
 
plt.plot(clusters_num, errors,marker='o') 
plt.show()

kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(data)
lables = kmeans_model.labels_
error = kmeans_model.inertia_
predict = kmeans_model.predict(data)
   
X = data[:,0:1]
y = data[:,1:2]

plt.scatter(X,y,s = 50)
plt.xlim(0,16)
plt.ylim(0,16)
plt.title('original data')
plt.show()

plt.scatter(data[predict == 0, 0], data[predict == 0, 1], s = 50, c = 'r')
plt.scatter(data[predict == 1, 0], data[predict == 1, 1], s = 50, c = 'b')
plt.scatter(data[predict == 2, 0], data[predict == 2, 1], s = 50, c = 'g')
plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], s = 150, c = 'y')
plt.xlim(0,16)
plt.ylim(0,16)
plt.title('data clustring')
plt.show()


