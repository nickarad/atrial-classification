# import tensorflow as tf 
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import labels as lb
# from tensorflow import keras
import matplotlib.pyplot as plt 
  
def load_data():
    dir = 'afpdbCSV/'
    records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.csv') != -1)]
    records.sort() 
    records = records[0:200]
    print(records)
    y = lb.get_labels()
    x = np.zeros((200,1280,2))
    j = 0
    for r in records:
        ecg = pd.read_csv(dir + r, index_col=0)
        ecg = ecg.values.tolist()
        i = 0
        for t in ecg:
            # print(i,t[1])
            x[j][i][0] = t[0]
            x[j][i][1] = t[1]
            # print(i,x[j][i][0],x[j][i][1])
            i = i+1

        j = j + 1

    return y,x

# y,x = load_data()
# np.save('x_data.npy', x)
# np.save('y_data.npy', y)
# print(x[199])
# print(y[199])


   

