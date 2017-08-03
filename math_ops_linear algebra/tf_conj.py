import tensorflow as tf

"""tf.conj(x, name=None)
功能：返回x的共轭复数。"""

a = tf.constant([1 + 2j, 2 - 3j])
z = tf.conj(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[1.-2.j  2.+3.j]
