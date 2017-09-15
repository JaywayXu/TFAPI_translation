"""def concat(values, axis, name="concat"):
解释：这个函数的作用是沿着concat_dim维度，去重新串联value，组成一个新的tensor。
tf.concat（axis, [list1, list2]）用于合并两个迭代器（比如列表）。axis表示合并的方向。0表示竖直合并，1表示水平
使用例子："""

import tensorflow as tf

sess = tf.Session()
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
d1 = tf.concat([t1, t2], 0)
d2 = tf.concat([t1, t2], 1)
print(sess.run(d1))
print(sess.run(tf.shape(d1)))  # [4 3]
print(sess.run(d2))
print(sess.run(tf.shape(d2)))  # [2 6]

# output
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
#
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

# tips
"""从直观上来看，我们取的concat_dim的那一维的元素个数肯定会增加。比如，上述例子中的d1的第0维增加了，而且d1.shape[0] = t1.shape[0]+t2.shape[0]。
输入参数：
  ● concat_dim: 一个零维度的Tensor，数据类型是int32。
  ● values: 一个Tensor列表，或者一个单独的Tensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个重新串联之后的Tensor。"""

