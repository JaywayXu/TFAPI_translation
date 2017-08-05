import tensorflow as tf

"""tf.argmin(input, axis=None, name=None, dimension=None)
功能：返回沿axis维度最小值的下标。"""

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], tf.float64)
z1 = tf.argmin(a, axis=0)
z2 = tf.argmin(a, axis=1)

sess = tf.Session()
print(sess.run(z1))
print(sess.run(z2))
sess.close()

# z1==>[0 0 0 0]
# z2==>[0 0 0]
