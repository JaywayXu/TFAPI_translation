import tensorflow as tf

"""tf.squared_difference(x,y,name=None)
功能：计算(x-y)(x-y)。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[-1, 0, 2]], tf.float64)
y = tf.constant([[2, 3, 4, ]], tf.float64)
z = tf.squared_difference(x, y)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[9. 9. 4.]]
