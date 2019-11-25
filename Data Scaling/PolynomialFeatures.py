import numpy as np
from sklearn.preprocessing import PolynomialFeatures

'''
PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)

'''

data = [[1,5],
        [2,6],
        [3,7],
        [4,8]]

print('original data')
print(np.matrix(data))
print("---------------------------------------------")

dataScaling = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True )
new_data = dataScaling.fit_transform(data)

print('data after scaling')
print(new_data)
