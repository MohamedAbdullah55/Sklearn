import numpy as np
from sklearn.preprocessing import StandardScaler

'''
StandardScaler(copy=True, with_mean=True, with_std=True)

'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))
print("---------------------------------------------")

dataScaling = StandardScaler(copy=True, with_mean=True, with_std=True)
new_data = dataScaling.fit_transform(data)

print('original data mean')
print(dataScaling.mean_)
print("---------------------------------------------")

print('original data scale')
print(dataScaling.scale_)
print("---------------------------------------------")

print('data after scaling')
print(new_data)
