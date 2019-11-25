import numpy as np
from sklearn.model_selection import LeaveOneOut

'''
يترك عنصر واحد للاختبار و الباقي للتدريب

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

y = np.array([[1],
     [0],
     [1],
     [1],
     [0],
     [1],
     [1],
     [0],
     [0],
     [1]])

loo = LeaveOneOut()
print('number of splits = ', str(loo.get_n_splits(X)))

print("----------------------------------------------------------")

folds = loo.split(X)

for train_index , test_index in folds:
    print('train : ',train_index,' test : ',test_index)
    print('X_train \n ',X[train_index])
    print('X_test  \n ',X[test_index])
    print('y_train \n ',y[train_index])
    print('y_test  \n ',y[test_index])
    print("----------------------------------------------------------")