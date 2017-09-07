"""tf.nn.l2_loss(t, name=None)
解释：这个函数的作用是利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半，
output = sum(t ** 2) / 2
"""

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.arange(1, 7).reshape(2, 3), dtype=tf.float32)
output = tf.nn.l2_loss(input_data)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input_data))
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
"""
[[ 1.  2.  3.]
 [ 4.  5.  6.]]
45.5
[]
"""

"""输入参数：
  ● t: 一个Tensor。数据类型必须是一下之一：float32，float64，int64，int32，uint8，int16，int8，complex64，qint8，quint8，qint32。虽然一般情况下，数据维度是二维的。但是，数据维度可以取任意维度。
  ● name: 为这个操作取个名字。
输出参数：
一个 Tensor ，数据类型和 t 相同，是一个标量。"""
