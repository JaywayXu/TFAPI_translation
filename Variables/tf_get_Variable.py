"""tf.get_variable(name, shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，初始化的方式有以下几种：
tf.constant_initializer：常量初始化函数
tf.random_normal_initializer：正态分布
tf.truncated_normal_initializer：截取的正态分布
tf.random_uniform_initializer：均匀分布
tf.zeros_initializer：全部是0
tf.ones_initializer：全是1
tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a1 = tf.get_variable(name='a1', shape=[2, 3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
a3 = tf.get_variable(name='a3', shape=[2, 3], initializer=tf.ones_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(a3))

    # [[0.42299312 - 0.25459203 - 0.88605702]
    #  [0.22410156  1.34326422 - 0.39722782]]
    # [1.]
    # [[1.  1.  1.]
    #  [1.  1.  1.]]

    # 注意：不同的变量之间不能有相同的名字，除非你定义了variable_scope，这样才可以有相同的名字。
