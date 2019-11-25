import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

'''
LogisticRegression(penalty=’l2’, dual=False, tol=0.0001,
                    C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                    random_state=None, solver=’warn’,max_iter=100, multi_class=’warn’, 
                    verbose=0, warm_start=False, n_jobs=None)
'''

print("Logistic Regression applaied on iris data set")

print("----------------------------------------------------------------------")

#load data
dataSet = load_iris()
X       = dataSet.data
y       = dataSet.target

print('X dimentions = ',X.shape)
print('y dimentions = ',y.shape)

#data scaling
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X          = normalizer.fit_transform(X)

#data spliting
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.30, random_state = 55, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.10, random_state = 55, shuffle=True)

#model bulding
LogisticRegression_1 = LogisticRegression(penalty='l2', solver='newton-cg', random_state = 55) 
LogisticRegression_2 = LogisticRegression(penalty='l2', solver='sag', random_state = 55) 
LogisticRegression_3 = LogisticRegression(penalty='l2', solver='lbfgs', random_state = 55) 
LogisticRegression_4 = LogisticRegression(penalty='l1', solver='liblinear', random_state = 55)
LogisticRegression_5 = LogisticRegression(penalty='l2', solver='liblinear', random_state = 55)
LogisticRegression_6 = LogisticRegression(penalty='l1', solver='saga', random_state = 55)
LogisticRegression_7 = LogisticRegression(penalty='l2', solver='saga', random_state = 55) 

#model run
LogisticRegression_1.fit(X_train,y_train)
LogisticRegression_2.fit(X_train,y_train)
LogisticRegression_3.fit(X_train,y_train)
LogisticRegression_4.fit(X_train,y_train)
LogisticRegression_5.fit(X_train,y_train)
LogisticRegression_6.fit(X_train,y_train)
LogisticRegression_7.fit(X_train,y_train)

#model score
print('Logistic Regression model 1 Train Score is : ' , LogisticRegression_1.score(X_train, y_train))
print('Logistic Regression model 1 Valid Score is : ' , LogisticRegression_1.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 2 Train Score is : ' , LogisticRegression_2.score(X_train, y_train))
print('Logistic Regression model 2 Valid Score is : ' , LogisticRegression_2.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 3 Train Score is : ' , LogisticRegression_3.score(X_train, y_train))
print('Logistic Regression model 3 Valid Score is : ' , LogisticRegression_3.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 4 Train Score is : ' , LogisticRegression_4.score(X_train, y_train))
print('Logistic Regression model 4 Valid Score is : ' , LogisticRegression_4.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 5 Train Score is : ' , LogisticRegression_5.score(X_train, y_train))
print('Logistic Regression model 5 Valid Score is : ' , LogisticRegression_5.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 6 Train Score is : ' , LogisticRegression_6.score(X_train, y_train))
print('Logistic Regression model 6 Valid Score is : ' , LogisticRegression_6.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

print('Logistic Regression model 7 Train Score is : ' , LogisticRegression_7.score(X_train, y_train))
print('Logistic Regression model 7 Valid Score is : ' , LogisticRegression_7.score(X_valid, y_valid))

print("----------------------------------------------------------------------")

#model validation
y_valid_pred_1 = LogisticRegression_1.predict(X_valid)
y_valid_pred_2 = LogisticRegression_2.predict(X_valid)
y_valid_pred_3 = LogisticRegression_3.predict(X_valid)
y_valid_pred_4 = LogisticRegression_4.predict(X_valid)
y_valid_pred_5 = LogisticRegression_5.predict(X_valid)
y_valid_pred_6 = LogisticRegression_6.predict(X_valid)
y_valid_pred_7 = LogisticRegression_7.predict(X_valid)

#model metrics on validation set
validation_cm_for_model_1 = confusion_matrix(y_valid,y_valid_pred_1)
validation_cm_for_model_2 = confusion_matrix(y_valid,y_valid_pred_2)
validation_cm_for_model_3 = confusion_matrix(y_valid,y_valid_pred_3)
validation_cm_for_model_4 = confusion_matrix(y_valid,y_valid_pred_4)
validation_cm_for_model_5 = confusion_matrix(y_valid,y_valid_pred_5)
validation_cm_for_model_6 = confusion_matrix(y_valid,y_valid_pred_6)
validation_cm_for_model_7 = confusion_matrix(y_valid,y_valid_pred_7)

for i in range(1,8):
    print("confusion_matrix on validation set for model ", i , "\n")
    if i == 1:
        print(validation_cm_for_model_1)
    elif i == 2:   
        print(validation_cm_for_model_2)
    elif i == 3:   
        print(validation_cm_for_model_3)
    elif i == 4:   
        print(validation_cm_for_model_4)
    elif i == 5:   
        print(validation_cm_for_model_5)
    elif i == 6:   
        print(validation_cm_for_model_6)
    elif i == 7:   
        print(validation_cm_for_model_7)    
        
    print("----------------------------------------------------------------------")    

    

#model testing
y_test_pred_1 = LogisticRegression_1.predict(X_test)
y_test_pred_2 = LogisticRegression_2.predict(X_test)
y_test_pred_3 = LogisticRegression_3.predict(X_test)
y_test_pred_4 = LogisticRegression_4.predict(X_test)
y_test_pred_5 = LogisticRegression_5.predict(X_test)
y_test_pred_6 = LogisticRegression_6.predict(X_test)
y_test_pred_7 = LogisticRegression_7.predict(X_test)

#model metrics on validation set
cm_for_model_1 = confusion_matrix(y_test,y_test_pred_1)
cm_for_model_2 = confusion_matrix(y_test,y_test_pred_2)
cm_for_model_3 = confusion_matrix(y_test,y_test_pred_3)
cm_for_model_4 = confusion_matrix(y_test,y_test_pred_4)
cm_for_model_5 = confusion_matrix(y_test,y_test_pred_5)
cm_for_model_6 = confusion_matrix(y_test,y_test_pred_6)
cm_for_model_7 = confusion_matrix(y_test,y_test_pred_7)

for i in range(1,8):
    print("confusion_matrix on tesy set for model ", i , "\n")
    if i == 1:
        print(cm_for_model_1)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_1,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_1))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_1))
        sns.heatmap(cm_for_model_1, center = True)
        plt.show()
    elif i == 2:   
        print(cm_for_model_2)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_2,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_2))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_2))
        sns.heatmap(cm_for_model_2, center = True)
        plt.show()
    elif i == 3:   
        print(cm_for_model_3)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_3,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_3))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_3))
        sns.heatmap(cm_for_model_3, center = True)
        plt.show()
    elif i == 4:   
        print(cm_for_model_4)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_4,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_4))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_4))
        sns.heatmap(cm_for_model_4, center = True)
        plt.show()
    elif i == 5:   
        print(cm_for_model_5)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_5,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_5))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_5))
        sns.heatmap(cm_for_model_5, center = True)
        plt.show()
    elif i == 6:   
        print(cm_for_model_6)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_6,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_6))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_6))
        sns.heatmap(cm_for_model_6, center = True)
        plt.show()
    elif i == 7:   
        print(cm_for_model_7)
        print("model ",i," f1 score = ",f1_score(y_test,y_test_pred_7,average='micro'))
        print("model ",i," MAE score = ",mean_absolute_error(y_test,y_test_pred_7))
        print("model ",i," MSE score = ",mean_squared_error(y_test,y_test_pred_7))
        sns.heatmap(cm_for_model_7, center = True)
        plt.show()
        
    print("----------------------------------------------------------------------")

print('Used classes : ',LogisticRegression_1.classes_)






