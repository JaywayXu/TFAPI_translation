"""tf.nn.moments(x, axes, name=None)
解释：这个函数的作用是计算 x 的均值和方差。
沿着 axes 维度，计算 x 的均值和方差。如果 x 是一维的，并且 axes = [0] ，那么就是计算整个向量的均值和方差。
如果，我们取 axes = [0, 1, 2] (batch, height, width)，那么我们就是计算卷积的全局标准化。
如果只是计算批处理的标准化，那么我们取 axes = [0] (batch) """

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.arange(1, 7).reshape(2, 3), dtype=tf.float32)
mean, variance = tf.nn.moments(input_data, [0])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input_data))
    print(sess.run(mean))
    print(sess.run(tf.shape(mean)))
"""
输入参数：
  ● x: 一个Tensor。
  ● axes: 一个整型的数组，确定计算均值和方差的维度 。
  ● name: 为这个操作取个名字。
输出参数：
两个 Tensor ，分别是均值 mean 和方差 variance 。"""

"""计算卷积神经网络某层的的mean和variance
假定我们需要计算数据的形状是 [batchsize, height, width, kernels]，熟悉CNN的都知道，
这个在tensorflow中太常见了，例程序如下："""

# img = tf.Variable(tf.random_normal([128, 32, 32, 64]))
img = tf.Variable(np.arange(1, 145).reshape(3, 4, 4, 3), dtype=tf.float32)
axis = list(range(len(img.get_shape()) - 1))
mean_1, variance_1 = tf.nn.moments(img, axis)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("the shape of the img", sess.run(tf.shape(img)))
    print("the value of the axis is ", axis)
    # [3 4 4 3]
    # [0, 1, 2]
    """the shape of the img [3 4 4 3]
    the value of the axis is  [0, 1, 2]"""
    print(sess.run(img))
    print("the value of the mean_1", sess.run(mean_1))  # 表示在通道上的平均值
    print("the shape of the mean_1", sess.run(tf.shape(mean_1)))
    print("the variance of the img", sess.run(variance_1))
    print("the shape of the variance_1", sess.run(tf.shape(variance_1)))
"""
the value of the mean_1 [ 71.5  72.5  73.5]
the shape of the mean_1 [3]
the variance of the img [ 1727.25  1727.25  1727.25]
the shape of the variance_1 [3]
"""
"""
其实很简单，可以这么理解，一个batch里的3个图，经过一个3 kernels卷积层处理，
得到了3*3个图，再针对每一个kernel所对应的3个图，求它们所有像素的mean和variance，
因为总共有3个kernels，输出的结果就是一个一维长度3的数组啦！


"""
