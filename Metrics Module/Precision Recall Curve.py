from sklearn.metrics import precision_recall_curve

'''
precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)
Compute precision-recall pairs for different probability thresholds
Note: this implementation is restricted to the binary classification task.

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("precision_recall_fscore_support")
PRC = precision_recall_curve(y_tarin,y_pred)
print(PRC)

print("--------------------------------------------------------------")

PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_tarin,y_pred)

print('Precision Value is : ', PrecisionValue)
print('Recall Value is : ', RecallValue)
print('Thresholds Value is : ', ThresholdsValue)
