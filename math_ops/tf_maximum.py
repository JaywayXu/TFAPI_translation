import tensorflow as tf

"""tf.maximum(x,y,name=None)
功能：计算x,y对应位置元素较大的值。支持广播模式
输入：x，y为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`类型。"""
x = tf.constant([[0.2, 0.8, -0.7], [-1, -3, -5]], tf.float64)
y = tf.constant([[0.2, 0.5, -0.3]], tf.float64)
z = tf.maximum(x, y)

sess = tf.Session()
print(sess.run(z))
sess.close()
# [[ 0.2  0.8 -0.3]
#  [ 0.2  0.5 -0.3]]
