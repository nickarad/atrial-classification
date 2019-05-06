import wfdb
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
import neurokit as nk
from biosppy.signals import ecg
import pandas as pd



def load_data():
    # Create Traiining dataset
    train = 'training/'
    # test = 'test/'
    x_train = np.array([])
    y_train = np.array([])
    records = [f for f in listdir(train) if isfile(join(train, f)) if(f.find('.csv') != -1)]
    records.sort() 
    counter = 0
    for r in records:
        if r[0] == 'n':
            y_train = np.append(y_train,0)
        elif r[0] =='p':
            y_train = np.append(y_train,1)

        test = pd.read_csv(train + r,index_col=0)
        test = test.values.tolist() # --> convert test dataframe to list
        x = np.array([])
        for t in test:
            # print(t[1])
            x = np.append(x,t[0])
        # print(x)
        x_train = np.append(x_train,x)
        counter = counter + 1
    x_train = x_train.reshape(-1, 1280)
    y_train = y_train.astype(int)
    # ==================================================================================
    # Create testing
    testing = 'test/'
    x_test = np.array([])
    y_test = np.array([])
    records = [f for f in listdir(testing) if isfile(join(testing, f)) if(f.find('.csv') != -1)]
    records.sort() 
    counter = 0
    for r in records:
        if r[0] == 'n':
            y_test = np.append(y_test,0)
        elif r[0] =='p':
            y_test = np.append(y_test,1)

        test = pd.read_csv(testing + r,index_col=0)
        test = test.values.tolist() # --> convert test dataframe to list
        x = np.array([])
        for t in test:
            # print(t[1])
            x = np.append(x,t[0])
        # print(x)
        x_test = np.append(x_test,x)
        counter = counter + 1
    x_test = x_test.reshape(-1, 1280)
    y_test = y_test.astype(int)

    return x_train,y_train,x_test,y_test


# x_train,y_train,x_test,y_test = load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# np.save('x_train.npy', x_train)
# np.save('y_train.npy', y_train)
# np.save('x_test.npy', x_test)
# np.save('y_test.npy', y_test)



