import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

'''
sklearn.linear_model.SGDClassifier(loss='hinge’, penalty=’l2’, alpha=0.0001,l1_ratio=0.15, fit_intercept=True,
                                   max_iter=None,tol=None, shuffle=True, verbose=0, epsilon=0.1,n_jobs=None,
                                   random_state=None, learning_rate='optimal’, eta0=0.0, power_t=0.5,
                                   early_stopping=False, validation_fraction=0.1,n_iter_no_change=5,
                                   class_weight=None,warm_start=False, average=False, n_iter=None)
'''

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 55, shuffle=True)

#model bulding and runing
num_of_iterations = []
accuracy = []
cost = []

for i in range(100,10000,100):
    
    sgd_classifier = SGDClassifier(loss='modified_huber', penalty='l1',learning_rate='optimal',random_state=33, max_iter=i)
    sgd_classifier.fit(X_train, y_train)
                   
    #model score
    #print('SGDClassifier Train Score is : ' , sgd_classifier.score(X_train, y_train))

    #model testing
    y_test_pred = sgd_classifier.predict(X_test)
    
    num_of_iterations.append(i)
    #accuracy.append(f1_score(y_test,y_test_pred,average='micro'))
    cost.append(mean_absolute_error(y_test,y_test_pred))
    #model metrics on validation set
    #cm = confusion_matrix(y_test,y_test_pred)
    #print(cm)
    #print("f1 score = ",f1_score(y_test,y_test_pred,average='micro'))
    #print("MAE score = ",mean_absolute_error(y_test,y_test_pred))
    #print("MSE score = ",mean_squared_error(y_test,y_test_pred))
    #print("Iterations = ",sgd_classifier.n_iter_)
    
    #sns.heatmap(cm)
    #plt.show()

plt.plot(num_of_iterations,cost,marker='o')
plt.xlabel = "num_of_iterations"
plt.ylabel = "cost"
plt.show()

