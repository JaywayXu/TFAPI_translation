import tensorflow as tf

"""tf.imag(input, name=None)
功能：返回虚数部分。
输入：`complex64`,`complex128`类型。"""

a = tf.constant([1 + 2j, 2 - 3j])
z = tf.imag(a)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[2.  -3.]
