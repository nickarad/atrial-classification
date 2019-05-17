import tensorflow as tf
from tensorflow import keras
import ecg_plot as pl
import numpy as np
# ================== Load weights from checkpoint and re-evaluate ===========================
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
new_model = keras.models.load_model('best.h5')
new_model.summary()
loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy {:5.2f}%".format(100*acc))

# ===========================================================================================

# ================================ Make predictions==========================================
predictions = new_model.predict(x_test)

for x in range(0,20):
    num = x
    # num = 18
    print(num)
    print("prediction:", np.argmax(predictions[num]))
    print("real value:", y_test[num])
    pl.ecg_plot(x_test[num])


# ============================================================================================

