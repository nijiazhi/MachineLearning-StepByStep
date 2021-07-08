# coding:utf8

import numpy as np

a1 = [1, 2, 3]
m1 = np.array(a1)
print(m1.shape)
print()

a1 = [[1, 2, 3]]
m1 = np.array(a1)
print(m1.shape)
print()


a1 = [[1, 2, 3], [2, 4, 6]]
m1 = np.array(a1)
print(m1.shape)
print()

tensor_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
m2 = np.array(tensor_list)
print(m2.shape)
print()

tensor_list2 = [[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]], [[13, 14, 15, 16]], [[17, 18, 19, 20]]]
m3 = np.array(tensor_list2)
print(m3.shape)
print()
