import tensorflow as tf


a = tf.constant([False, True, False], dtype=tf.bool)
a1 = tf.constant([True, True, False], dtype=tf.bool)
b = tf.logical_not(a)  # 逻辑非
b1 = tf.logical_and(a, a1)  # 逻辑与
b2 = tf.logical_or(a, a1)  # 逻辑或
b3 = tf.logical_xor(a, a1)  # 异或如果a、b两个值不相同，则异或结果为1。如果a、b两个值相同，异或结果为0。
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(b1))
    print(sess.run(b2))
    print(sess.run(b3))
sess.close()
# [ True False  True]
# [False  True False]
# [ True  True False]
# [ True False False]
