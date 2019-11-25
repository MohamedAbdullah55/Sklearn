from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif

'''
SelectPercentile(score_func=<function f_classif>, percentile=10)

'''
breastData = load_breast_cancer()
X = breastData.data
y = breastData.target

print('original data features number = ',str(X.shape[1]),' feature')

FeatureSelectionMethod = SelectPercentile(score_func=chi2, percentile=10)
new_X = FeatureSelectionMethod.fit_transform(X,y)

print('new data features number = ',str(new_X.shape[1]),' feature')

print('selected features are')
print(FeatureSelectionMethod.get_support())


