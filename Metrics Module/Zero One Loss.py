from sklearn.metrics import zero_one_loss

'''

zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)
Zero-one classification loss.
If normalize is True, return the fraction of misclassifications (float), 
else it returns the number of misclassifications(int). The best performance is 0.

'''
y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

print("zero_one_loss")
ZOL = zero_one_loss(y_tarin,y_pred)
print(ZOL)
