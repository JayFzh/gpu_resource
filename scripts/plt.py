import tkinter
import  matplotlib
# matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import numpy as np

# data = np.array([[10., 30., 19., 22.],
#                 [5., 18., 15., 20.],
#                 [4., 6., 3., 5.]])
# color_list = ['b', 'g', 'r']
# ax2 = plt.subplot(111)
# X = np.arange(data.shape[1])
# print(X)
# for i in range(data.shape[0]):#i表示list的索引值
#     ax2.bar(X, data[i],
#          width=0.2,
#          bottom = np.sum(data[:i], axis = 0),
#          color = color_list[i % len(color_list)],
#             alpha =0.5
#             )
# plt.savefig('zhifangtu.png',dpi=120,bbox_inches='tight')


names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
# plt.figure(figsize=(9, 3))
plt.subplot(2,2,1)
plt.bar(names, values)
plt.subplot(2,2,2)
plt.scatter(names, values)
plt.subplot(2,2,3)
plt.plot(names, values)
plt.subplot(2,2,4)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')


# data = [[5,25,50,20],
#                 [4,23,51,17],
#                 [6,22,52,19]]
# X = np.arange(4) #四类柱子
# plt.bar(X + 0.00, data[0], color = 'b', width = 0.25,label = "A")# 每一类柱子的第一个蓝色柱子
# plt.bar(X + 0.25, data[1], color = 'g', width = 0.25,label = "B")# X+0.25是因为一个柱子的width是0.25
# plt.bar(X + 0.5, data[2], color = 'r', width = 0.25,label = "C")
# plt.xticks(np.linspace(0,3,4),['1月','2月','3月','4月'],rotation=45)
# plt.legend(loc='upper left')
plt.savefig('picture/2.png',dpi=120,bbox_inches = 'tight')
# plt.show()

# print(['fusion{}_{}'.format(i+1,j+1) for i in range(0,2) for j in range(0,4)])
