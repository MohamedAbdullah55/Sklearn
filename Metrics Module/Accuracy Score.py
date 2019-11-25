from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

'''
accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
((TP + TN) / float(TP + TN + FP + FN))

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("confusion matrix")
cm = confusion_matrix(y_tarin,y_pred)
print(cm)

print("--------------------------------------------------------------")

print("accuracy score with normalize=True")
AS = accuracy_score(y_tarin, y_pred,  normalize=True)
print(AS)

print("--------------------------------------------------------------")

print("accuracy score with normalize=False")
AS = accuracy_score(y_tarin, y_pred,  normalize=False)
print(AS)