from sklearn.metrics import roc_auc_score

'''
roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,
max_fpr=None)
Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
Note: this implementation is restricted to the binary classification task or multilabel classification task in label
indicator format.

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("roc_auc_score")
RAS = roc_auc_score(y_tarin,y_pred)
print(RAS)

