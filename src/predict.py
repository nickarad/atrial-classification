import tensorflow as tf
from tensorflow import keras
import ecg_plot as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
# ================== Load weights from checkpoint and re-evaluate ===========================
x_data = np.load('../data/x_data.npy')
y_data = np.load('../data/y_data.npy')
train = 0.7
size = 200
x_test = x_data[int(train * size):size]
y_test = y_data[int(train * size):size]
print(x_data.shape)
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
new_model = keras.models.load_model('../models/best.h5')
new_model.summary()
loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy {:5.2f}%".format(100*acc))

# ================================ Make predictions==========================================
predictions = new_model.predict(x_test)

# for x in range(0,20):
#     num = x
#     # num = 18
#     print(num)
#     print("prediction:", np.argmax(predictions[num]))
#     print("real value:", y_test[num])
#     pl.ecg_plot(x_test[num])

# ************************************* Metrics ***************************************************
y_pred = np.array([])
for x in range(0,len(predictions)):
	y_pred = np.append(y_pred,np.argmax(predictions[x]))
	
y_pred = y_pred.astype(int)

# Precision:
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and 
# fp the number of false positives. The precision is intuitively the ability of the classifier 
# not to label as positive a sample that is negative.
# The best value is 1 and the worst value is 0.

precision = precision_score(y_test, y_pred, average='micro') # micro calculates metrics globally by counting the total true positives, false negatives and false positives.
print("Precision:", precision)

# Recall (aka Sensitivity):
# Recall is the ratio of the correctly +ve labeled by our program to all who have atrial fibrillation in reality.
# Recall answers the following question: Of all the people who have atrial fibrillation, how many of those we correctly predict?
# Recall = TP/(TP+FN)
recall = recall_score(y_test, y_pred, average='micro') # micro calculates metrics globally by counting the total true positives, false negatives and false positives.
print("Recall:", recall) 

# Specificity:
# Specificity is the correctly -ve labeled by the program to all who are healthy in reality.
# Specifity answers the following question: Of all the people who are healthy, how many of those did we correctly predict?
# Specificity = TN/(TN+FP)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
tn = float(tn)
specificity = tn / (tn+fp)
print("Specificity:", specificity) 



