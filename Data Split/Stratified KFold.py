import numpy as np
from sklearn.model_selection import StratifiedKFold

'''
StratifiedKFold(n_splits=’warn’, shuffle=False, random_state=None)
This cross-validation object is a variation of KFold that returns stratified folds. 
The folds are made by preserving the percentage of samples for each class.
using for classification purpose
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

#print("All X data")
#print(X)
#
#print("----------------------------------------------------------")
#
#print("All y data")
#print(y)
#
#print("----------------------------------------------------------")

skf = StratifiedKFold(n_splits=2,shuffle=True,random_state=55)
print('number of splits = ', str(skf.get_n_splits(X)))

print("----------------------------------------------------------")

folds = skf.split(X,y)

for train_index , test_index in folds:
    print('train : ',train_index,' test : ',test_index)
    print('X_train \n ',X[train_index])
    print('X_test  \n ',X[test_index])
    print('y_train \n ',y[train_index])
    print('y_test  \n ',y[test_index])
    print("----------------------------------------------------------")