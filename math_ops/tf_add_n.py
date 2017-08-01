"""tf.add_n(inputs,name=None)
功能：将所有输入的tensor进行对应位置的加法运算
输入：inputs：一组tensor，必须是相同类型和维度。
add_n和add的区别在于add_n不支持广播操作，即add_n相加的对象必须保持相同的类型和维度"""
import tensorflow as tf

x = tf.constant([[1, 2, -3]], tf.float64)
y = tf.constant([[2, 3, 4]], tf.float64)
z = tf.constant([[1, 4, 3]], tf.float64)
xyz = [x, y, z]
z = tf.add_n(xyz)
# z = tf.add_n(x, y, z)  # 这样写是不对的，inputs必须是一个tensor而不能是多个~
# TypeError: add_n() takes from 1 to 2 positional arguments but 3 were given
sess = tf.Session()
print(sess.run(z))
sess.close()
# [[ 4.  9.  4.]]
