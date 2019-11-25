from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

'''
SelectKBest(score_func=<function f_classif>, k=10)
Select features according to the k highest scores.

'''

breastData = load_breast_cancer()
X = breastData.data
y = breastData.target

print('original data features number = ',str(X.shape[1]),' feature')

FeatureSelectionMethod = SelectKBest(score_func=chi2, k = 5)
new_X = FeatureSelectionMethod.fit_transform(X,y)

print('new data features number = ',str(new_X.shape[1]),' feature')

print('selected features are')
print(FeatureSelectionMethod.get_support())