import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
MinMaxScaler(feature_range=(0, 1), copy=True)

'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))
print("---------------------------------------------")

dataScaling = MinMaxScaler(feature_range=(-0.5,0.5), copy=True)
new_data = dataScaling.fit_transform(data)

print('original data max')
print(dataScaling.data_max_)
print("---------------------------------------------")

print('original data min')
print(dataScaling.data_min_)
print("---------------------------------------------")

print('original data range')
print(dataScaling.data_range_)
print("---------------------------------------------")

print('data after scaling')
print(new_data)
