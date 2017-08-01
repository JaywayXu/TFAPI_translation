import tensorflow as tf

"""tf.sign(x,name=None)
功能：求x的符号，x>0,则y=1;x<0则y=-1;x=0则y=0。
输入：x,为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。"""

x = tf.constant([[1.1, 0, -3]], tf.float64)
z = tf.sign(x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1. 0. -1.]]
