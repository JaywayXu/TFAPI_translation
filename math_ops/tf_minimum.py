import tensorflow as tf

"""tf.minimum(x,y,name=None)
功能：计算x,y对应位置元素较小的值。并且支持广播模式
输入：x，y为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`类型。"""

x = tf.constant([[0.2, 0.8, -0.7], [3, 4, 5]], tf.float64)
y = tf.constant([[0.2, 0.5, -0.3]], tf.float64)
z = tf.minimum(x, y)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[ 0.2  0.8 -0.3]
#       [ 3.   4.   5. ]] 这里的[3, 4, 5]是同一个维度的，这样写是没有错的
# 但是如果第二个维度不是三维度的而是[3， 4， 5， 6]这样的形式，这样就会报错
