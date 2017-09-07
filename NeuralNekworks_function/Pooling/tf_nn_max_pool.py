"""池化操作是利用一个矩阵窗口在输入张量上进行扫描，并且将每个矩阵窗口中的值通过取最大值，平均值或者其他操作来减少元素个数。
每个池化操作的矩阵窗口大小是由 ksize 来指定的，并且根据步长参数 strides 来决定移动步长。
比如，如果 strides 中的值都是1，那么每个矩阵窗口都将被使用。如果 strides 中的值都是2，那么每一维度上的矩阵窗口都是每隔一个被使用。
以此类推。
更具体的输出结果是：
output[i] = reduce( value[ strides * i: strides * i + ksize ] )
输出数据维度是：
shape(output) = (shape(value) - ksize + 1) / strides"""

"""def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
解释：这个函数的作用是计算池化区域中元素的最大值。
使用例子：
"""

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
# 此处的数据表示右10个批次的数据,6高度6宽度3个通道数
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
# 此处的fileter表示卷积核,为[2高度,2宽度,3输入通道数.10输出通道数]

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
# 此时y输出的shape为(10,6,6,10)其中第一个参数表示10个批次,
# 第二个参数6表示feature map中的高度,第三个参数6表示宽度
# 第四个参数表示10个新的通道数.
output = tf.nn.max_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run(output)
    print(sess.run(tf.shape(y)))
    print(sess.run(tf.shape(output)))
    # [10  6  6 10]
    # [10  5  5 10]
"""输入参数：
  ● value: 一个四维的Tensor。数据维度是 [batch, height, width, channels]。数据类型是float32.
  ● ksize: 一个长度不小于4的整型数组。每一位上面的值对应于输入数据张量中每一维的窗口对应值。
    即对应于图像的宽和高的是ksize的第1和第2个参数这里表示对应图像窗口值为(2,2)
  ● strides: 一个长度不小于4的整型数组。该参数指定滑动窗口在输入数据张量每一维上面的步长。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
  ● data_format:表示数据的格式
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和value相同。"""
