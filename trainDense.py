import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import dense as dn
from tensorflow import keras
import data1D as data


# ************************************* Prepare Dataset ************************************************
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# -- Normalise data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# ************************************* Create Model ***************************************************
model = dn.create_model()
model.fit(x_train, y_train, epochs=9)
# -- model accurancy
val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
# ======================================================================================================

# ************************************* Make predictions ***************************************************
predictions = model.predict(x_test)
test = 0
print("prediction:", np.argmax(predictions[test]))
print("real value:", y_test[test])
# ======================================================================================================

# ************************************* Save Model ***************************************************
# model.summary()
# Save entire model
model.save('my_model.h5')
# ======================================================================================================