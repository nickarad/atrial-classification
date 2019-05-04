import tensorflow as tf 
from tensorflow import keras
import numpy as np
from os import listdir
from os.path import isfile, join
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
# print(x)
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
# print(y_train)
# print(y_test)
