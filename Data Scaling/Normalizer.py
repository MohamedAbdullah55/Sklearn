import numpy as np
from sklearn.preprocessing import Normalizer

'''
Normalizer(norm=’l2’, copy=True)

و هي مخصصة لتناول كل صف علي حدة في المصفوفات ثنائية الأبعاد 
تستخدم l1  لجعل مجموع كل صف هو القيمة العظمي 
تستخدم l2 لجعل جذر مجموع مربعات كل صف هو القيمة العظمي 
تستخدم max   لجعل القيمة العظمي في كل صف هي القيمة العظمي
 
'''

data = [[1,2],
        [2985,95874],
        [100,500],
        [-58552,-55224]]

print('original data')
print(np.matrix(data))

print("---------------------------------------------")

dataScaling_1 = Normalizer(norm='l1')
dataScaling_2 = Normalizer(norm='l2')
dataScaling_3 = Normalizer(norm='max')

new_data_1 = dataScaling_1.fit_transform(data)
new_data_2 = dataScaling_2.fit_transform(data)
new_data_3 = dataScaling_3.fit_transform(data)

print('data after scaling using norm=l1')
print(new_data_1)

print("---------------------------------------------")

print('data after scaling using norm=l2')
print(new_data_2)

print("---------------------------------------------")

print('data after scaling using norm=max')
print(new_data_3)
