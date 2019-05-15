import numpy as np
import matplotlib.pyplot as plt 
# import neurokit as nk

x = np.load('x_train.npy')
signal = x[199,:,0]

fs = 128
Time=np.linspace(0, len(signal)/fs, num=len(signal))
print(Time)

# plt.figure(1)
# plt.title('Record n01')
plt.plot(Time,signal,'-', lw=1.6)
# plt.grid(True,which='both', color='0.65', linestyle='-')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()