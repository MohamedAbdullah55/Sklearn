import numpy as np
from sklearn.model_selection import KFold

'''
KFold(n_splits=’warn’, shuffle=False, random_state=None)

'''

X = np.array([[1,11],
     [2,12],
     [3,13],
     [4,14],
     [5,15],
     [6,16],
     [7,17],
     [8,18],
     [9,19],
     [10,20]])

y = np.array([[11],
     [22],
     [33],
     [44],
     [55],
     [66],
     [77],
     [88],
     [99],
     [1010]])

#print("All X data")
#print(X)
#
#print("----------------------------------------------------------")
#
#print("All y data")
#print(y)
#
#print("----------------------------------------------------------")

kf = KFold(n_splits=5,shuffle=True)
print('number of splits = ', str(kf.get_n_splits(X)))

print("----------------------------------------------------------")

folds = kf.split(X)

for train_index , test_index in folds:
    print('train : ',train_index,' test : ',test_index)
    print('X_train \n ',X[train_index])
    print('X_test  \n ',X[test_index])
    print("----------------------------------------------------------")