import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error

'''
SVR(kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001,C=1.0, 
    epsilon=0.1, shrinking=True, cache_size=200, verbose=False,max_iter=-1)

'''

#load data
dataSet = load_boston()
X       = dataSet.data
y       = dataSet.target

print('X dimentions = ',X.shape)
print('y dimentions = ',y.shape)

#data scaling
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X          = normalizer.fit_transform(X)

#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 55, shuffle=True)

#model bulding
degress = []
cost = []
for i in range(12):
    
    print("At degree = ",i)
    svr_model = SVR(kernel='poly', C=50, epsilon=0.1, degree=i)
    
    #model run
    svr_model.fit(X_train,y_train)
    
    #model score
    print('SVR Train Score is : ' , svr_model.score(X_train, y_train))
    print('SVR Test Score is : ' , svr_model.score(X_test, y_test))
    
    #model testing
    y_test_pred= svr_model.predict(X_test)
    
    mae = mean_absolute_error(y_test,y_test_pred)
    mse = mean_squared_error(y_test,y_test_pred)
    
    degress.append(i)
    cost.append(mae)

    print("MAE score = ",mae)
    print("MSE score = ",mse)
    print("---------------------------------------------")
    

plt.plot(degress,cost, marker='o') 
plt.xlabel = 'Degree'
plt.ylabel = 'Cost' 
plt.show()  

