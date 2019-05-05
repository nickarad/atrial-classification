import tensorflow as tf 
from tensorflow import keras
import numpy as np
from os import listdir
from os.path import isfile, join
import createmodel as crm
import warnings

warnings.filterwarnings("ignore")

# dir = 'afpdbCSV/'
# records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.csv') != -1)]
# records.sort() 
# records = records[0:200]
# print(records)

# =========================================== load Dataset ===========================================================
x = np.load('x_data.npy')
y = np.load('y_data.npy')
print(x.shape)
# print(y)
# =========================================== Create dataset ===========================================================
x_train = x[0:80]
x_train = np.append(x_train,x[100:180])
y_train = y[0:80]
y_train = np.append(y_train,y[100:180])
x_test = x[80:100]
x_test = np.append(x_test,x[180:200])
y_test = y[80:100]
y_test = np.append(y_test,y[180:200])
x_train = x_train.reshape(-1, 1280,2)
x_test = x_test.reshape(-1, 1280,2)
print(x_train.shape)
print(y_train)
# print(y_test)

# =========================================== Create Model ===========================================================
model = crm.create_model()
model.fit(x_train, y_train, epochs=3)
# -- model accurancy
val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

# =========================================== Make predictions ===========================================================
predictions = model.predict(x_test)
test = 2
print("prediction:", np.argmax(predictions[test]))
print("real value:", y_test[test])

# =========================================== Save Model ===========================================================
model.save('my_model.h5')
