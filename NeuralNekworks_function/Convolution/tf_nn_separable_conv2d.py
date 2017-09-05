"""这个卷积是为了避免卷积核在全通道的情况下进行卷积，这样非常浪费时间。
使用这个API，你将应用一个二维的卷积核，在每个通道上，以深度 channel_multiplier 进行卷积。
先利用 depthwise_filter,将ID的通道数映射到 ID * DM 的通道数上面，之后从 ID * DM的通道数映射到OD的通道数上面，
这也就是上面说的深度 channel_multiplier 对应于 DM。
具体公式如下：
output[b, i, j, k] = sum_{di, dj, q, r]
    input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    depthwise_filter[di, dj, q, r] *
    pointwise_filter[0, 0, q * channel_multiplier + r, k]
strides 只是仅仅控制 depthwise convolution 的卷积步长，因为 pointwise convolution 的卷积步长是确定的 [1, 1, 1, 1] 。
注意，必须有 strides[0] = strides[3] = 1。在大部分处理过程中，卷积核的水平移动步数和垂直移动步数是相同的，
即 strides = [1, stride, stride, 1]。"""

"""
它与tf.nn.conv2d类似,对于规模较大的模型,它可在不牺牲准确率的前提下实现训练的加速.
"""

"""
def separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate=None, name=None,
                     data_format=None):
"""
import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
pointwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)
# out_channels >= channel_multiplier * in_channels
y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y))
    print(sess.run(tf.shape(y)))  # [10  6  6 20]
"""输入参数：
  ● input: 一个Tensor。数据维度是四维 [batch, in_height, in_width, in_channels]。
  ● depthwise_filter: 一个Tensor。数据维度是四维 [filter_height, filter_width, in_channels, channel_multiplier]。其中，in_channels 的卷积深度是 1。
  ● pointwise_filter: 一个Tensor。数据维度是四维 [1, 1, channel_multiplier * in_channels, out_channels]。其中，pointwise_filter 是在 depthwise_filter 卷积之后的混合卷积。
  ● strides: 一个长度是4的一维整数类型数组，每一维度对应的是 input 中每一维的对应移动步数，比如，strides[1] 对应 input[1] 的移动步数。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个四维的Tensor，数据维度为 [batch, out_height, out_width, out_channels]。
异常：
  ● 数值异常: 如果 channel_multiplier * in_channels > out_channels ，那么将报错。"""
