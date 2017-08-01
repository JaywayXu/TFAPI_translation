"""tf.multiply(x, y, name = None)
解释：这个函数返回x与y逐元素相乘的结果。
使用例子："""

import tensorflow as tf
# 数组和数的相乘
a = tf.constant([1, 2])
b = tf.constant(2)
c = tf.multiply(a, b)
sess = tf.Session()
print(sess.run(c))  # [2 4]
sess.close()
# 两个数组相乘
a0 = tf.constant([1, 2])
b0 = tf.constant([3, 4])
c0 = tf.multiply(a0, b0)
sess = tf.Session()
print(sess.run(c0))  # [3 8]
sess.close()
#两个高维数组相乘
"""高维数组相乘使用matmul函数，如果使用函数multiply则会报错"""
a1 = tf.constant([[1, 2], [3, 4], [5, 6]])
b1 = tf.constant([[1, 2, 3], [4, 5, 6]])
c1 = tf.matmul(a1, b1)
sess = tf.Session()
print(sess.run(c1))
# [[ 9 12 15]
#  [19 26 33]
#  [29 40 51]]
sess.close()
# c2 = tf.multiply(a1, b1) 这条语句会报错
# print(sess.run(c2))
# sess.close()
"""输入参数：
  ● x: 一个Tensor，数据类型是必须是以下之一：float32，float64，int8，int16，int32，complex64，int64。
  ● y: 一个Tensor，数据类型必须和x相同。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和x相同。"""
"""
PS:
declaration->multiply
    def multiply(x, y, name=None):
    return gen_math_ops._mul(x, y, name)
     "2016-12-30",
    "`tf.mul(x, y)` is deprecated, please use `tf.multiply(x, y)` or `x * y`")

    declaration->gen_math_ops._mul
       def _mul(x, y, name=None):
          Returns x * y element-wise.
          *NOTE*: `Mul` supports broadcasting. More about broadcasting
          result = _op_def_lib.apply_op("Mul", x=x, y=y, name=name)
  return result
"""