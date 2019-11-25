from sklearn.metrics import precision_recall_fscore_support

'''
metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=
                                        None, pos_label=1, average=
                                        None, warn_for=(‘precision’,
                                        ’recall’, ’f-score’), sample_
                                        weight=None)

'''
y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("precision_recall_fscore_support")
PRFS = precision_recall_fscore_support(y_tarin,y_pred,average='micro')
print(PRFS)

print("--------------------------------------------------------------")