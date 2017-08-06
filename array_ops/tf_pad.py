"""tf.pad(input, paddings, name = None)
解释：这个函数的作用是向input中按照paddings的格式填充0。
paddings是一个整型的Tensor，数据维度是[n, 2]，其中n是input的秩。
对于input的中的每一维D，paddings[D, 0]表示增加多少个0在input之前，
paddings[D, 1]表示增加多少个0在input之后。
举个例子，假设paddings = [[1, 1], [2, 2]]和input的数据维度是[2,2]，
那么最后填充完之后的数据维度如下：
[1+2+1,2+2+2]

填充之后的数据维度
也就是说，最后的数据维度变成了[4,6]。"""
"""def pad(tensor, paddings, mode="CONSTANT", name=None):
 pad(t, paddings, "CONSTANT") ==> [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 2, 3, 0, 0],
                                    [0, 0, 4, 5, 6, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]

  pad(t, paddings, "REFLECT") ==> [[6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1],
                                   [6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1]]

  pad(t, paddings, "SYMMETRIC") ==> [[2, 1, 1, 2, 3, 3, 2],
                                     [2, 1, 1, 2, 3, 3, 2],
                                     [5, 4, 4, 5, 6, 6, 5],
                                     [5, 4, 4, 5, 6, 6, 5]] 
"""


import tensorflow as tf

sess = tf.Session()
t = tf.constant([[[3, 3], [2, 2]]])
print(sess.run(tf.shape(t)))
# [1 2 2]
paddings = tf.constant([[3, 3], [1, 1], [2, 2]])
# 表示在最内层中前面增加2个0，后面增加2个0，上面一层增加一层全零，下面一层增加一层全零
# 然后是在最外层增加三个相同shape的全0层
print(paddings.shape)
# (3, 2)
print(sess.run(tf.pad(t, paddings)).shape)
# 关于shape的解释[3+1+3, 1+2+1, 2+2+2]=[7, 4, 6]
# (7, 4, 6)
print(sess.run(tf.pad(t, paddings)))
"""
[[[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]

 [[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]

 [[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]

 [[0 0 0 0 0 0]
  [0 0 3 3 0 0]
  [0 0 2 2 0 0]
  [0 0 0 0 0 0]]  关键是这一层，要重点理解

 [[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]

 [[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]

 [[0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]
  [0 0 0 0 0 0]]]"""
"""输入参数：
  ● input: 一个Tensor。
  ● paddings: 一个Tensor，数据类型是int32。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和input相同。"""
