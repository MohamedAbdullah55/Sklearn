from sklearn.metrics import classification_report

'''
metrics.classification_report(y_true, y_pred, labels=None, target_names=None,
sample_weight=None, digits=2, output_dict=False)
Build a text report showing the main classification metrics

'''

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("classification_report")
CR = classification_report(y_tarin,y_pred)
print(CR)