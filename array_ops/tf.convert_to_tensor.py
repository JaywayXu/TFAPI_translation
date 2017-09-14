# tf.convert_to_tensor用于将不同数据变成张量：比如可以让数组变成张量、也可以让列表变成张量。
import tensorflow as tf
import numpy as np

A = list([1, 2, 3])
B = np.array([1, 2, 3])
C = tf.convert_to_tensor(A)
D = tf.convert_to_tensor(B)

with tf.Session() as sess:
    print(type(A))
    print(type(B))
    print(type(C))
    print(type(D))
    # < class 'list'>
    #
    # < class 'numpy.ndarray'>
    #
    # < class 'tensorflow.python.framework.ops.Tensor'>
    #
    # < class 'tensorflow.python.framework.ops.Tensor'>
