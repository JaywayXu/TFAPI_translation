"""卷积操作的空间含义定义如下：如果输入数据是一个四维的 input ，卷积操作的步长stride是一个四维数组.
数据维度是 [batch, in_height, in_width, ...]，卷积核也是一个四维的卷积核，数据维度是 [filter_height, filter_width, ...] ，那么：
shape(output) = [batch,
                (in_height - filter_height + 1) / strides[1],
                (in_width - filter_width + 1) / strides[2],
                ...]
...中的数据表示通道数,例如对于图像就表示像素点的RGB值
output[b, i, j, :] = sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, ...] * filter[di, dj, ...]
因为，input 数据是一个四维的，每一个通道上面是一个向量 input[b, i, j, :] 。对于 conv2d ，这些向量将会被卷积核 filter[di, dj, :, :] 相乘而产生一个新的向量。

对于 depthwise_conv_2d ，每个标量分量 input[b, i, j, k] 将在 k 个通道上面独立的被卷积核 filter[di, dj, k] 进行卷积操作，然后把所有得到的向量进行连接组合成一个新的向量。"""

"""
def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None):
           
解释：这个函数的作用是对一个四维的输入数据 input 和四维的卷积核 filter 进行操作，然后对输入数据进行一个二维的卷积操作，最后得到卷积之后的结果。
给定的输入张量的维度是 [batch, in_height, in_width, in_channels],
卷积核张量的维度是 [filter_height, filter_width, in_channels, out_channels],参数分别表示卷积核的[高度,宽度,输入通道数.输出通道数]
具体卷积操作如下：
  ● 将卷积核的维度转换成一个二维的矩阵形状 [filter_height * filter_width * in_channels, output_channels]
  ● 对于每个批处理的图片，我们将输入张量转换成一个临时的数据维度 [batch, out_height, out_width, filter_height * filter_width * in_channels] 。
  ● 对于每个批处理的图片，我们右乘以卷积核，得到最后的输出结果。
更加具体的表示细节为：
output[b, i, j, k] =
           sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                     filter[di, dj, q, k]
注意，必须有 strides[0] = strides[3] = 1。在大部分处理过程中，卷积核的水平移动步数和垂直移动步数是相同的，即 strides = [1, stride, stride, 1] 。"""
import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 1), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(y))
    print(sess.run(tf.shape(y)))  # [10  6  6  1]
"""输入参数：
  ● input: 一个Tensor。数据类型必须是float32或者float64。表示输入.
  ● filter: 一个Tensor。数据类型必须是input相同,表示卷积核.
  ● strides: 一个长度是4的一维整数类型数组，步长.每一维度对应的是 input 中每一维的对应移动步数，比如，strides[1] 对应 input[1] 的移动步数。
  ● padding: 一个字符串，取值为 SAME 或者 VALID 。
    SAME:卷积输出与输入的尺寸相同,这里在计算如何跨越图像时,不用考虑滤波器的尺寸.
    VALID:计算卷积核如何在图像上跨越时,需要考虑滤波器的尺寸,卷积核不能超过图像的尺寸.
  ● use_cudnn_on_gpu: 一个可选布尔值，默认情况下是 True 。
  ● name: （可选）为这个操作取一个名字。
  ● data_format:用于修改输入的格式,该参数可取为"NHWC"或"NCHW",默认值是"NHWC"用于指定输入和输出数据的格式,
    当取默认格式"NHWC"数据存储的顺序为[batch,in_height,in_width,in_channels]
    数据格式    N:批数据中的张量数目,即batch_size
              H:每个批数据中张量的高度
              W:每个批数据中张量的宽度
              C:每个批数据中张量的通道数
输出参数：
  ● 一个Tensor，数据类型是 input 相同。"""
