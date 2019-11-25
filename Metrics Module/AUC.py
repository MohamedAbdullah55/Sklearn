from sklearn.metrics import roc_curve
from sklearn.metrics import auc

'''
auc(x, y, reorder=’deprecated’)
Compute Area Under the Curve (AUC) using the trapezoidal rule

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

fprValue, tprValue, thresholdsValue = roc_curve(y_tarin,y_pred)

print("auc")
AUC = auc(fprValue,tprValue)
print(AUC)

