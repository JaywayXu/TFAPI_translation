import tensorflow as tf

"""tf.diag_part(input,name=None)
功能：返回对角阵的对角元素。
输入：tensor,且维度必须一致。"""

a = tf.constant([[1, 5, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
z = tf.diag_part(a)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[1,2,3,4]
