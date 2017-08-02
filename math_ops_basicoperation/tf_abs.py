"""tf.abs(x,name=None)
功能：求x的绝对值。
输入：x为张量或稀疏张量，可以为`float32`, `float64`,  `int32`, `int64`类型。"""
import tensorflow as tf

x = tf.constant([[1.1, 2, -3]], tf.float64)
z = tf.abs(x)
sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[1.1 2. 3.]]
