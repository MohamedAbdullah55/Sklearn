from sklearn.metrics import roc_curve

'''
roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
drop_intermediate=True)
Compute Receiver operating characteristic (ROC)
Note: this implementation is restricted to the binary classification task.

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("roc_curve")
RC = roc_curve(y_tarin,y_pred)
print(RC)

print("--------------------------------------------------------------")

fprValue, tprValue, thresholdsValue = roc_curve(y_tarin,y_pred)
print('fpr Value  : ', fprValue)
print('tpr Value  : ', tprValue)
print('thresholds Value  : ', thresholdsValue)
