"""归一化(标准化)的目标之一在于将输入保持在一个可接受的范围内
例如将输入归一化到[0.0,1.0]区间内使输入中所有可能的分量归一化为一个大于等于0.0小于等于1.0的值"""
"""
tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
解释：这个函数的作用是利用 L2 范数对指定维度 dim 进行标准化。
比如，对于一个一维的张量，指定维度 dim = 0，那么计算结果为：
output = x / sqrt( max( sum( x ** 2 ) , epsilon ) )
假设 x 是多维度的，那么标准化只会独立的对维度 dim 进行，不会影响到别的维度。"""
import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.arange(1, 7).reshape(2, 3), dtype=tf.float32)
output = tf.nn.l2_normalize(input_data, dim=0)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(tf.shape(input_data)))
    print(sess.run(input_data))
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
    # [2 3]
    # [[1.  2.  3.]
    #  [4.  5.  6.]]
    # [[0.24253564  0.37139067  0.44721359]
    #  [0.97014254  0.92847669  0.89442718]]
    # [2 3]
"""输入参数：
  ● x: 一个Tensor。
  ● dim: 需要标准化的维度。
  ● epsilon: 一个很小的值，确定标准化的下边界。如果 norm < sqrt(epsilon)，那么我们将使用 sqrt(epsilon) 进行标准化。
  ● name: （可选）为这个操作取一个名字。
输出参数：
一个 Tensor ，数据维度和 x 相同。"""
