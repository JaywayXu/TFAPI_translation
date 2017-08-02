import tensorflow as tf

"""tf.matrix_band_part(input,num_lower,num_upper,name=None)
功能：复制一个矩阵，并将规定带之外的元素置为0。
     假设元素坐标为（m，n），则in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                                          (num_upper < 0 || (n-m) <= num_upper)。
    band（m,n）=in_band(m,n)*input(m,n)。
    特殊情况：
          tf.matrix_band_part(input, 0, -1) ==> 上三角阵.
          tf.matrix_band_part(input, -1, 0) ==> 下三角阵.
          tf.matrix_band_part(input, 0, 0) ==> 对角阵.
输入：num_lower:如果为负，则结果右上空三角阵;
     num_upper:如果为负，则结果左下为空三角阵。"""
a = tf.constant([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]])
z = tf.matrix_band_part(a, 1, -1)  # 左下角空三角阵
# z==>[[0 1 2 3]
#      [-1 0 1 2]
#      [0 -1 0 1]
#      [0 0 -1 0]]
z1 = tf.matrix_band_part(a, 1, -2)  # 只要位置是负的话就行，与负数的数值无关
# [[ 0  1  2  3]
#  [-1  0  1  2]
#  [ 0 -1  0  1]
#  [ 0  0 -1  0]]
z2 = tf.matrix_band_part(a, -1, 1)  # 右上角空三角阵
# [[ 0  1  0  0]
#  [-1  0  1  0]
#  [-2 -1  0  1]
#  [-3 -2 -1  0]]
z3 = tf.matrix_band_part(a, 0, -1)
# [[0 1 2 3]
#  [0 0 1 2]
#  [0 0 0 1]
#  [0 0 0 0]]
z4 = tf.matrix_band_part(a, -1, 0)
# [[ 0  0  0  0]
#  [-1  0  0  0]
#  [-2 -1  0  0]
#  [-3 -2 -1  0]]
z5 = tf.matrix_band_part(a, 0, 0)
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]

sess = tf.Session()
print(sess.run(z))
print(sess.run(z1))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
print(sess.run(z5))
sess.close()

