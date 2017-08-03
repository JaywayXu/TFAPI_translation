import tensorflow as tf

"""tf.complex(real, imag, name=None)
功能：将实数转化为复数。
输入：real，imag：float32或float64。"""

real = tf.constant([1, 2], dtype=tf.float64)
imag = tf.constant([3, 4], dtype=tf.float64)
z = tf.complex(real, imag)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[1.+3.j  2.+4.j]
