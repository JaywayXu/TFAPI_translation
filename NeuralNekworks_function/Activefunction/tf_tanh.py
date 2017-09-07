"""tf.tanh(x, name = None)
解释：这个函数的作用是计算 x 的 tanh 函数。具体计算公式为 ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )。
tanh和tf.sigmoid非常接近,且与后者具有类似的优缺点,tf.sigmoid和tf.tanh的主要区别在于后者的值域为[-1.0,1.0]
在一些特定的网络架构中,能够输出负值的能力十分有用.
但是注意tf.tanh值域的中间点为0.0,当网络中的下一层期待输入为负值或者为0.0时,这将引发一系列问题.
"""

import tensorflow as tf

a = tf.constant([[-1.0, -2.0], [1.0, 2.0], [0.0, 0.0]])
sess = tf.Session()
print(sess.run(tf.tanh(a)))
# [[-0.76159418 -0.96402758]
#  [ 0.76159418  0.96402758]
#  [ 0.          0.        ]]
"""输入参数：
  ● x: 一个Tensor。数据类型必须是float，double，int32，complex64，int64或者qint32。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，如果 x.dtype != qint32 ，那么返回的数据类型和x相同，否则返回的数据类型是 quint8 。"""
