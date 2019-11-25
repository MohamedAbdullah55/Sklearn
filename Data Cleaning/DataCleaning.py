from sklearn.impute import SimpleImputer
import numpy as np

'''
impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)

'''

data = [[1,2,np.nan],
        [3,np.nan,1],
        [5,np.nan,0],
        [np.nan,4,6 ],
        [5,0,np.nan],
        [4,5,5]]

print("Data Before imputation")
print("---------------------------------------------")
print(np.matrix(data))

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(data)
imputedData = imputer.transform(data)

print("Data After imputation")
print("---------------------------------------------")
print(imputedData)