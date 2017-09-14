# tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
import tensorflow as tf

A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A))
    sess.run(tf.assign(A, 10))
    print(sess.run(A))
    # 0.0
    # 10.0
