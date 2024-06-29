
import numpy as np

data01 = np.load('data_set/train_outputs.npy')
data02 = np.load('data_set/train_inputs.npy')
data03 = np.load('data_set/val_inputs.npy')
data04 = np.load('data_set/val_outputs.npy')

print(data01.shape)  # 打印数组的内容
print(data02.shape)
print(data03.shape)
print(data04.shape)
