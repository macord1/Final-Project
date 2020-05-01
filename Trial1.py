import numpy as np
import matplotlib.pyplot as plt

file = open('NII_Data.csv')
line = next(file).strip('\n').split(',')
fs = 2.23221E4
dt = 1 / fs
train_time = 5
train_data = []
test_data = []
for i in range(0, int(train_time * fs)):
    train_data.append(line[i])
for i in range(int(train_time * fs) + 1, len(line)):
    test_data.append(line[i])
Train_t = np.arange(0.0, 5.0 - dt, dt)
Test_t = np.arange(5.0, 25.0, dt)
print(len(train_data))
print(len(Train_t))
# plt.plot(Train_t, train_data)
# plt.show()
