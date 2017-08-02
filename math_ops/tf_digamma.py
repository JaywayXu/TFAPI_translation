import tensorflow as tf

"""tf.digamma(x,name=None)
功能：计算lgamma的导数，即gamma'/gamma。即(（x-1）!)'/(x-1)!
输入：x，y为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[1, 2, 3]], tf.float64)
z = tf.digamma(x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[-0.57721566 0.42278434 0.92278434]]
