"""变量减"""
# tf.assign_sub(A, new_number): 这个函数的功能主要是把A的值减去new_number

import tensorflow as tf

A = tf.Variable(tf.constant(2.0), dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A))
    sess.run(tf.assign_sub(A, 10))
    print(sess.run(A))
    # 2.0
    # -8.0
