"""def avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
解释：这个函数的作用是计算池化区域中元素的平均值。"""

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='VALID')
output = tf.nn.avg_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run(output))
    print(sess.run(tf.shape(y)))
    print(sess.run(tf.shape(output)))
    # [10  5  5 10]
    # [10  4  4 10]
"""输入参数：
  ● value: 一个四维的Tensor。数据维度是 [batch, height, width, channels]。数据类型是float32，float64，qint8，quint8，qint32。
  ● ksize: 一个长度不小于4的整型数组。每一位上面的值对应于输入数据张量中每一维的窗口对应值。
  ● strides: 一个长度不小于4的整型数组。该参数指定滑动窗口在输入数据张量每一维上面的步长。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
  ● data_format:表示数据的不同结构
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和value相同。"""
