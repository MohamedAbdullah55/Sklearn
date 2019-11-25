from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

'''
SelectFromModel(estimator, threshold=None, prefit=False,norm_order=1, max_features=None)
Meta-transformer for selecting features based on importance weights.

'''

breastData = load_breast_cancer()
X = breastData.data
y = breastData.target

print('original data features number = ',str(X.shape[1]),' feature')

model = RandomForestClassifier(n_estimators=10)
FeatureSelectionMethod = SelectFromModel(estimator=model)
new_X = FeatureSelectionMethod.fit_transform(X,y)

print('new data features number = ',str(new_X.shape[1]),' feature')

print('selected features are')
print(FeatureSelectionMethod.get_support())