"""各种各样的div除法形式"""
"""tf.truediv(x,y,name=None)
功能：对应位置元素的除法运算。（使用python3除法算法，又叫真除，结果为浮点数，推荐使用tf.divide）
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。"""


"""tf.floordiv(x,y,name=None)
功能：对应位置元素的地板除法运算。返回不大于结果的最大整数
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。"""
import tensorflow as tf

x = tf.constant([[2, 4, -1]], tf.int64)  # float类型运行结果一致，只是类型为浮点型
y = tf.constant([[3, 3, 3]], tf.int64)
z = tf.floordiv(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# [[ 0  1 -1]]
x = tf.constant([[2, 4, -1]], tf.float32)  # 注意x和y的数据类型应该要一样才可以
y = tf.constant([[3, 3, 3]], tf.float32)
z = tf.floordiv(x, y)
sess = tf.Session()
print(sess.run(z))
# [[ 0.  1. -1.]]
sess.close()


""" 1.8 tf.realdiv(x,y,name=None)
# 功能：对应位置元素的实数除法运算。实际情况与divide结果没区别，
# 输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`,
# `complex64`, `complex128`, `string‘类型。"""
# 例：
x = tf.constant([[2 + 1j, 4 + 2j, -1 + 3j]], tf.complex64)
y = tf.constant([[3 + 3j, 3 + 1j, 3 + 2j]], tf.complex64)
z = tf.realdiv(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0.50000000-0.16666667j 1.39999998+0.2j 0.23076922+0.84615386j]]


"""tf.truncatediv(x,y,name=None)
# 功能：对应位置元素的截断除法运算，获取整数部分。
# 输入：x,y具有相同尺寸的tensor，可以为`uint8`, `int8`, `int16`, `int32`, `int64`,类型。(只能为整型，浮点型等并未注册，和手册不符)"""
# 例：
x = tf.constant([[2, 4, -7]], tf.int64)
y = tf.constant([[3, 3, 3]], tf.int64)
z = tf.truncatediv(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0 1 -2]]
# x = tf.constant([[2, 4, -7]], tf.float64)  # 如果此处为浮点数类型则会出现未注册的错误
# y = tf.constant([[3, 3, 3]], tf.float64)
# z = tf.truncatediv(x, y)
# sess = tf.Session()
# print(sess.run(z))
# sess.close()


"""1.10 tf.floor_div(x,y,name=None)
功能：对应位置元素的地板除法运算。（和tf.floordiv运行结果一致，只是内部实现方式不一样）
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`,
`complex64`, `complex128`, `string‘类型。"""
