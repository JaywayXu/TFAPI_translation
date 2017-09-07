"""局部响应归一化层
def lrn(input, depth_radius=None, bias=None, alpha=None, beta=None,name=None):
解释：这个函数的作用是计算局部数据标准化。
输入的数据 input 是一个四维的张量，但该张量被看做是一个一维的向量（ input 的最后一维作为向量），
向量中的每一个元素都是一个三维的数组（对应 input 的前三维）。
向量的每一个元素都是独立的被标准化的。具体数学形式如下：
sqr_sum[a, b, c, d] =
    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum ** beta)"""

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(1, 2, 3, 4), dtype=tf.float32)
output = tf.nn.local_response_normalization(input_data)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input_data))
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
"""输入参数：
  ● input: 一个Tensor。数据维度是四维的，数据类型是 float32 。
  ● depth_radius: （可选）一个整型，默认情况下是 5 。
  ● bias: （可选）一个浮点型，默认情况下是 1。一个偏移项，为了避免除0，一般情况下我们取正值。
  ● alpha: （可选）一个浮点型，默认情况下是 1。一个比例因子，一般情况下我们取正值。
  ● beta: （可选）一个浮点型，默认情况下是 0.5。一个指数。
  ● name: （可选）为这个操作取一个名字。
输出参数：
一个 Tensor ，数据类型是 float32 。"""
