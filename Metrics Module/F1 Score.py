from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

'''
f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
F1 = 2 * (precision * recall) / (precision + recall)

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

print("recall_score")
rs = recall_score(y_tarin,y_pred,average='micro')
print(rs)

print("--------------------------------------------------------------")

print("f1 score")
f1 = f1_score(y_tarin,y_pred,average='micro')
print(f1)

print("--------------------------------------------------------------")