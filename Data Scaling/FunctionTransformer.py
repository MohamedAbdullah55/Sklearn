import numpy as np
from sklearn.preprocessing import FunctionTransformer

'''
FunctionTransformer(func=None, inverse_func=None, validate=
                    None, accept_sparse=False,
                    pass_y=’deprecated’,
                    check_inverse=True, kw_args=None,
                    inv_kw_args=None)

'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))
print("---------------------------------------------")

def transformFun(x):
    return x**2

dataScaling = FunctionTransformer(func=transformFun, validate=True)
new_data = dataScaling.fit_transform(data)

print('data after scaling')
print(new_data)
