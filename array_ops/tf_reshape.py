"""tf.reshape(tensor, shape, name = None)
解释：这个函数的作用是对tensor的维度进行重新组合。给定一个tensor，这个函数会返回数据维度是shape的一个新的tensor，但是tensor里面的元素不变。
如果shape是一个特殊值[-1]，那么tensor将会变成一个扁平的一维tensor。
如果shape是一个一维或者更高的tensor，那么输入的tensor将按照这个shape进行重新组合，但是重新组合的tensor和原来的tensor的元素是必须相同的。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(sess.run(data))
print(sess.run(tf.shape(data)))
# [2 2 3]
d = tf.reshape(data, [-1])
print(sess.run(d))
# [1 1 1 2 2 2 3 3 3 4 4 4]
d = tf.reshape(data, [3, 4])
print(sess.run(d))
# [[1 1 1 2]
#  [2 2 3 3]
#  [3 4 4 4]]
d = tf.reshape(data, [-1, 3, 2])
print(sess.run(d))
"""输入参数：
  ● tensor: 一个Tensor。
  ● shape: 一个Tensor，数据类型是int32，定义输出数据的维度。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和输入数据相同。"""
