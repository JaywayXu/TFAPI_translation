"""tf.tanh(x, name = None)
解释：这个函数的作用是计算 x 的 tanh 函数。具体计算公式为 ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )。"""

import tensorflow as tf

a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
sess = tf.Session()
print(sess.run(tf.tanh(a)))
"""输入参数：
  ● x: 一个Tensor。数据类型必须是float，double，int32，complex64，int64或者qint32。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，如果 x.dtype != qint32 ，那么返回的数据类型和x相同，否则返回的数据类型是 quint8 。"""
