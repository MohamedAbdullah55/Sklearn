import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

'''
ShuffleSplit(n_splits=10, test_size=’default’,train_size=None, random_state=None)
 يقوم بعمل اختيار عشوائي للتدريب و الاختبار حسب النسبة المعطاة مع مراعاة توزيع النسب 
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

s_shuf_s = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=55)
print('number of splits = ', str(s_shuf_s.get_n_splits(X)))

print("----------------------------------------------------------") 

folds = s_shuf_s.split(X,y)

for train_index , test_index in folds:
    print('train : ',train_index,' test : ',test_index)
    print('X_train \n ',X[train_index])
    print('X_test  \n ',X[test_index])
    print('y_train \n ',y[train_index])
    print('y_test  \n ',y[test_index])
    print("----------------------------------------------------------")