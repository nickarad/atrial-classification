import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import createmodel as crm
import pandas as pd
import ecg_plot as pl
from tensorflow.keras.callbacks import TensorBoard

# ************************************* Tensorboard ************************************************

NAME = "atrial"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# ************************************* Load Dataset ************************************************
x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')
train = 0.7
size = 200
x_train = x_data[0:int(train * size)]
y_train = y_data[0:int(train * size)]
x_test = x_data[int(train * size):size]
y_test = y_data[int(train * size):size]
print(x_data.shape)
print(y_data.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# -- Normalise data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# ************************************* Create Model ***************************************************
learn_rate = 0.01 # Define learning rate
ep = 15 # Number of epochs
batch = 16 # define batch size
model = crm.create_model(learn_rate)
history = model.fit(x_train, y_train, epochs=ep,validation_data=(x_test, y_test), callbacks=[tensorboard])
pd.DataFrame(history.history).to_csv(path_or_buf='logs/History.csv')
# -- model accurancy
val_loss1, val_acc1 = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
# print(val_loss1)  # model's loss (error)
print("Validation accuracy {:5.2f}%".format(100*val_acc1))
# print(val_acc1)  # model's accuracy

# ======================================================================================================
# Plot the model Accuracy graph (Ideally, it should be Logarithmic shape)
f, ax = plt.subplots(2, sharex=True)

ax[0].plot(history.history['acc'],'r',linewidth=2.0, label='Training Accuracy')
ax[0].plot(history.history['val_acc'],'b',linewidth=2.0, label='Testing Accuracy')
ax[0].legend(fontsize=10)
# ax[0].xlabel('Epochs ', fontsize=16)
# ax[0].ylabel('Accuracy', fontsize=16)
ax[0].set(xlabel='Epochs', ylabel='Accuracy')
ax[0].grid(b=True, which='major', color='#999999', linestyle='-',alpha=0.6)
ax[0].minorticks_on()
ax[0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.show()
# Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
ax[1].plot(history.history['loss'], 'g', linewidth=2.0, label='Training Loss')
ax[1].plot(history.history['val_loss'], 'y', linewidth=2.0, label='Testing Loss')
ax[1].legend(fontsize=10)
ax[1].set(xlabel='Epochs', ylabel='Loss Curves')
ax[1].grid(b=True, which='major', color='#999999', linestyle='-',alpha=0.6)
ax[1].minorticks_on()
ax[1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.show()
# ************************************* Make predictions ***************************************************
predictions = model.predict(x_test)
test = 19
print("prediction:", np.argmax(predictions[test]))
print("real value:", y_test[test])

# Plot prediction
signal = x_test[test]
pl.ecg_plot(signal)
# ======================================================================================================

# ************************************* Save Model ***************************************************
# model.summary()
# Save entire model
model.save('my_model.h5')
# ======================================================================================================