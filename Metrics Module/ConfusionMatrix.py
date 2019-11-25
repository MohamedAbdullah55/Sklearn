from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''

confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

'''

print("Binary Classification Case")

y_tarin = [1,0,1,0,1,0,1,0,1,0]
y_pred  = [1,1,0,1,0,1,0,1,0,1]

cm = confusion_matrix(y_tarin,y_pred)
print(cm)

plt.imshow(cm)
plt.colorbar()
plt.show()

sns.heatmap(cm, center = True)
plt.show()

print("--------------------------------------------------------")

print("Multi Classification Case")

y_tarin = [1,0,2,1,0,2,0,2,0,1,2,1,0,2,1,1,0]
y_pred  = [1,0,2,0,2,1,0,2,1,0,2,1,0,2,1,0,2]

cm = confusion_matrix(y_tarin,y_pred)
print(cm)

plt.imshow(cm)
plt.colorbar()
plt.show()

sns.heatmap(cm, center = True)
plt.show()

