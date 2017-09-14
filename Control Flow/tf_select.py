# 是一个布尔类型的变量，比如True，也可以是一个表达式，返回值是True或者False。
# 这个变量可以是一个也可以是一个列表，就是很多个True或者False组成的列表
# 如果是True，返回p2，反之返回p3
# 如果是一个列表的对比，那就是维度要一样，也就是参数的维度相同，那么p1的第一个元素是True，返回的对象的第一个值就是p2中的第一个值，反之是p3中的第一个值，依此类推
# 在tensorflow1.0+的版本中已经被取代为tf.where

import tensorflow as tf
import numpy as np

A = 3
B = tf.convert_to_tensor([1, 2, 3, 4])
C = tf.convert_to_tensor([1, 1, 1, 1])
D = tf.convert_to_tensor([0, 0, 0, 0])

with tf.Session() as sess:
    print(sess.run(tf.where(A > 1, 'A', 'B')))
    print(sess.run(tf.where(False, 'A', 'B')))
    print(sess.run(tf.where(B > 2, C, D)))
# b'A'
# b'B'
# [0 0 1 1]