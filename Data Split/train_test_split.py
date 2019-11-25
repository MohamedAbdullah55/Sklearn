import numpy as np
from sklearn.model_selection import train_test_split

X = [[1,11],
     [2,12],
     [3,13],
     [4,14],
     [5,15],
     [6,16],
     [7,17],
     [8,18],
     [9,19],
     [10,20]]

y = [[11],
     [22],
     [33],
     [44],
     [55],
     [66],
     [77],
     [88],
     [99],
     [1010]]

print("All X data")
print(np.matrix(X))

print("----------------------------------------------------------")

print("All y data")
print(np.matrix(y))

print("----------------------------------------------------------")

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.30, random_state=55, shuffle =True)

print("X_train data")

print(np.matrix(X_train))

print("----------------------------------------------------------")

print("X_test data")
print(np.matrix(X_test))

print("----------------------------------------------------------")
