"""展示各种对应元素取余数的方法"""
import tensorflow as tf

"""tf.mod(x,y,name=None)
功能：对应位置元素的除法取余运算。若x和y只有一个小于0，则计算‘floor（x/y）*y+mod(x,y)’。
输入：x,y具有相同尺寸的tensor，可以为`float32`, `float64`,  `int32`, `int64`类型。"""

x = tf.constant([[2.1, 4.1, -1.1], [5.1, 5.2, 5.3]], tf.float64)
y = tf.constant([[3, 3, 3]], tf.float64)
z = tf.mod(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[ 2.1  1.1  1.9] [ 2.1  2.2  2.3]]


"""tf.truncatemod(x,y,name=None)
功能：对应位置元素的截断除法取余运算。
输入：x,y具有相同尺寸的tensor，可以为float32`, `float64`,  `int32`, `int64`类型"""

x = tf.constant([[2.1, 4.1, -1.1]], tf.float64)
y = tf.constant([[3, 3, 3]], tf.float64)
z = tf.truncatemod(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[2.1 1.1 -1.1]]


"""tf.floormod(x,y,name=None)
# 功能：对应位置元素的地板除法取余运算。
# 输入：x,y具有相同尺寸的tensor，可以为float32`, `float64`,  `int32`, `int64`类型。"""
x = tf.constant([[2.1, 4.1, -1.1]], tf.float64)
y = tf.constant([[3, 3, 3]], tf.float64)
z = tf.floormod(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[2.1 1.1 1.9]]
