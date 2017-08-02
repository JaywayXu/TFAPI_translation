"""矩阵的迹
在线性代数中，一个n×n矩阵A的主对角线（从左上方至右下方的对角线）上各个元素的总和被称为矩阵A的迹"""
import tensorflow as tf

"""tf.trace(x,name=None)
功能：返回矩阵的迹。
输入：tensor"""
a = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
z = tf.trace(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[15 42]
