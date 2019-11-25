import numpy as np
from sklearn.preprocessing import Binarizer

'''
Binarizer(threshold=0.0, copy=True)

'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))
print("---------------------------------------------")

dataScaling = Binarizer(threshold=100, copy=True)
new_data = dataScaling.fit_transform(data)

print('data after scaling')
print(new_data)
