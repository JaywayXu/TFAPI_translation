"""tf.add(x, y, name = None)
解释：这个函数返回x与y逐元素相加的结果。
注意：tf.add操作支持广播形式，但是tf.add_n操作不支持广播形式。
使用例子："""

import tensorflow as tf
# 数字进行相加
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
sess = tf.Session()
print(sess.run(c))  # 5
sess.close()

# add数组相加加数是一个数
a0 = tf.constant([1, 2, 3])
b0 = tf.constant(2)
c0 = tf.add(a0, b0)
sess = tf.Session()
print(sess.run(c0))  # [3, 4, 5]
sess.close()

# 不同维度的数组相加
# a1 = tf.constant([1, 2, 2, 2])
# b1 = tf.constant([2, 2])
# c1 = tf.add(a1, b1)
# # ValueError: Dimensions must be equal, but are 4 and 2 for 'Add_2' (op: 'Add') with input shapes: [4], [2].
# sess = tf.Session()
# print(sess.run(c1))
# sess.close()

# 高维数组加上常数
a2 = tf.constant([[1, 2, 3], [2, 2, 2]])
b2 = tf.constant(2)
c2 = tf.add(a2, b2)
sess = tf.Session()
print(sess.run(c2))
# [[3 4 5]
#  [4 4 4]]
sess.close()

#高维数组相加的条件
a3 = tf.constant([[1, 2, 3], [2, 2, 2]])
b3 = tf.constant([3, 3, 3])
c3 = tf.add(a3, b3)
sess = tf.Session()
print(sess.run(c3))
# [[4 5 6]
#  [5 5 5]]
sess.close()

"""输入参数：
  ● x: 一个Tensor，数据类型是必须是以下之一：float32，float64，int8，int16，int32，complex64，int64。
  ● y: 一个Tensor，数据类型必须和x相同。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和x相同。"""