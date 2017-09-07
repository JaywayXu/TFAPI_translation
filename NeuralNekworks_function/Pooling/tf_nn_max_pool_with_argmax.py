"""def max_pool_with_argmax(input, ksize, strides, padding, Targmax=None,name=None):
解释：这个函数的作用是计算池化区域中元素的最大值和该最大值所在的位置。
因为在计算位置 argmax 的时候，我们将 input 铺平了进行计算，
所以，如果 input = [b, y, x, c]，那么索引位置是 ( ( b * height + y ) * width + x ) * channels + c 。
此函数适用于tensorflow-gpu版本"""
import numpy as np
import tensorflow as tf
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=tf.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
output, argmax = tf.nn.max_pool_with_argmax(input=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
"""输入参数：
  ● input: 一个四维的Tensor。数据维度是 [batch, height, width, channels]。数据类型是float32。
  ● ksize: 一个长度不小于4的整型数组。每一位上面的值对应于输入数据张量中每一维的窗口对应值。
  ● strides: 一个长度不小于4的整型数组。该参数指定滑动窗口在输入数据张量每一维上面的步长。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
  ● Targmax: 一个可选的数据类型： tf.int32或者tf.int64。默认情况下是 tf.int64 。
  ● name: （可选）为这个操作取一个名字。
输出参数：
一个元祖张量 (output, argmax)：
  ● output: 一个Tensor，数据类型是float32。表示池化区域的最大值。
  ● argmax: 一个Tensor，数据类型是Targmax。数据维度是四维的"""
