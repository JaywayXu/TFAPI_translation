import tensorflow as tf

"""tf.pow(x,y,name=None)
功能：计算x各元素的y次方。
输入：x，y为张量，可以为`float32`, `float64`, `int32`, `int64`,`complex64`,`complex128`类型。"""

x = tf.constant([[2, 3, 5], [2, 3, 5]], tf.float64)
y = tf.constant([[2, 3, 4]], tf.float64)
z = tf.pow(x, y)

sess = tf.Session()
print(sess.run(z))
sess.close()
"""[[   4.   27.  625.]
    [   4.   27.  625.]]"""
