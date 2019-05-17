import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import createmodel as crm
import ecg_plot as pl
from tensorflow.keras.callbacks import TensorBoard

# ************************************* Tensorboard ************************************************

NAME = "atrial"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# ************************************* Load Dataset ************************************************
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# -- Normalise data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# ************************************* Create Model ***************************************************
learn_rate = 0.02 # Define learning rate
ep = 9 # Number of epochs
batch = 64# define batch size
model = crm.create_model(learn_rate)
history = model.fit(x_train, y_train, epochs=ep,validation_data=(x_test, y_test), batch_size=batch, callbacks=[tensorboard])
# -- model accurancy
val_loss1, val_acc1 = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss1)  # model's loss (error)
print(val_acc1)  # model's accuracy

# ======================================================================================================
# Plot the model Accuracy graph (Ideally, it should be Logarithmic shape)
plt.plot(history.history['acc'],'r',linewidth=2.0, label='Training Accuracy')
plt.plot(history.history['val_acc'],'b',linewidth=2.0, label='Testing Accuracy')
plt.legend(fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()
# Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
plt.plot(history.history['loss'], 'g', linewidth=2.0, label='Training Loss')
plt.plot(history.history['val_loss'], 'y', linewidth=2.0, label='Testing Loss')
plt.legend(fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()
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