import numpy as np
from sklearn.preprocessing import MaxAbsScaler

'''
MaxAbsScaler(copy=True)
 مشابهة لـ normalizer لكن بالنسبة للعمود
 و ليس الصف حيث تجعل أكبر قيمة في كل عمود هي القيمة العظمي و تغير الباقيين علي اساسه
 
'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))

print("---------------------------------------------")

dataScaling = MaxAbsScaler(copy=True)

new_data = dataScaling.fit_transform(data)
print(new_data)
