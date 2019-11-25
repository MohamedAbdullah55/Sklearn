from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

'''
recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
(TP / float(TP + FN))   

0 => positive
1 => negative
 
'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("confusion matrix")
cm = confusion_matrix(y_tarin,y_pred)
print(cm)

print("--------------------------------------------------------------")

print("precision_score")
ps = precision_score(y_tarin,y_pred,average='micro')
print(ps)

print("--------------------------------------------------------------")