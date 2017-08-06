"""tf.tile(input, multm iples, name = None)
解释：这个函数的作用是通过给定的tensor去构造一个新的tensor。
所使用的方法是将input复制multiples次，输出的tensor的第i维有input.dims(i) * multiples[i]个元素，
input中的元素被复制multiples[i]次。比如，input = [a b c d], multiples = [2]，
那么tile(input, multiples) = [a b c d a b c d]。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[1, 2, 3, 4], [9, 8, 7, 6]])
d = tf.tile(data, [2, 3])
print(sess.run(d))
"""输入参数：
  ● input_: 一个Tensor，数据维度是一维或者更高维度。
  ● multiples: 一个Tensor，数据类型是int32，数据维度是一维，长度必须和input的维度一样。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和input相同。"""
