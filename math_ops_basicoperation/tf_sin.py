import tensorflow as tf

"""tf.sin(x,name=None)
功能：计算x的正弦值。
输入：x为张量，可以为`half`,`float32`, `float64`,  `complex64`, `complex128`类型。"""

x = tf.constant([[0, 1.5707963]], tf.float64)
z = tf.sin(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 1.]]
