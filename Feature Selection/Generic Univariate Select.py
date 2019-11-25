from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2, f_classif

'''
GenericUnivariateSelect(score_func=<functionf_classif>, mode=’percentile’,param=1e-05)

'''

breastData = load_breast_cancer()
X = breastData.data
y = breastData.target

print('original data features number = ',str(X.shape[1]),' feature')

FeatureSelectionMethod = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=5)
new_X = FeatureSelectionMethod.fit_transform(X,y)

print('new data features number = ',str(new_X.shape[1]),' feature')

print('selected features are')
print(FeatureSelectionMethod.get_support())