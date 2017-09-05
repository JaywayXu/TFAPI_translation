"""def depthwise_conv2d(input,filter,strides,padding,rate=None,name=None,data_format=None):
解释：这个函数也是一个卷积操作。
给定一个输入张量，数据维度是 [batch, in_height, in_width, in_channels] ，
一个卷积核的维度是 [filter_height, filter_width, in_channels, channel_multiplier] ，
在通道 in_channels 上面的卷积深度是 1 （我的理解是在每个通道上单独进行卷积），
depthwise_conv2d 函数将不同的卷积核独立的应用在 in_channels 的每个通道上（从通道 1 到通道 channel_multiplier ），
然后把所有的结果进行汇总。最后输出通道的总数是 in_channels * channel_multiplier 。
更加具体公式如下：
output[b, i, j, k * channel_multiplier + q] =
    sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                  filter[di, dj, k, q]
注意，必须有 strides[0] = strides[3] = 1。在大部分处理过程中，卷积核的水平移动步数和垂直移动步数是相同的，
即 strides = [1, stride, stride,1]。"""
# 当需要将一个卷积层的输出连接到另一个卷积层的输入时,可以使用这种卷积.

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)

y = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y))
    print(sess.run(tf.shape(y)))
    # [10  6  6 15]
"""输入参数：
  ● input: 一个Tensor。数据维度是四维 [batch, in_height, in_width, in_channels]。
  ● filter: 一个Tensor。数据维度是四维 [filter_height, filter_width, in_channels, channel_multiplier]。
  ● strides: 一个长度是4的一维整数类型数组，每一维度对应的是 input 中每一维的对应移动步数，比如，strides[1] 对应 input[1] 的移动步数。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
  ● use_cudnn_on_gpu: 一个可选布尔值，默认情况下是 True 。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个四维的Tensor，数据维度为 [batch, out_height, out_width, in_channels * channel_multiplier]。"""
