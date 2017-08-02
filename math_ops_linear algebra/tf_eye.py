import tensorflow as tf

"""tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)
功能：返回单位阵。
输入：num_rows:矩阵的行数;num_columns:矩阵的列数，默认与行数相等
            batch_shape:若提供值，则返回batch_shape的单位阵。"""

z1 = tf.eye(2, batch_shape=[2])  # 一个三维矩阵其中包含有两个2行2列的矩阵
z2 = tf.eye(2, batch_shape=[3])  # 一个三维矩阵其中包含有三个2行2列的矩阵
z3 = tf.eye(2, batch_shape=[2, 1])
z4 = tf.eye(2, batch_shape=[3, 2])
# 应该将整个特定形状的单位矩阵视为一个整体
sess = tf.Session()
print("This z1", sess.run(z1))
print("This z2", sess.run(z2))
print("This z3", sess.run(z3))
print("This z4", sess.run(z4))

sess.close()
"""
This z1 
[
[[ 1.  0.]
  [ 0.  1.]]

 [[ 1.  0.]
  [ 0.  1.]]
]  
This z2 
[
[[ 1.  0.]
  [ 0.  1.]]

 [[ 1.  0.]
  [ 0.  1.]]

 [[ 1.  0.]
  [ 0.  1.]]
]
This z3 
[
[[[ 1.  0.]
   [ 0.  1.]]]


 [[[ 1.  0.]
   [ 0.  1.]]]
]
This z4 
[
 [
   [[ 1.  0.]
   [ 0.  1.]]

   [[ 1.  0.]
   [ 0.  1.]]
 ]


 [
   [[ 1.  0.]
   [ 0.  1.]]

   [[ 1.  0.]
   [ 0.  1.]]
 ]


 [
   [[ 1.  0.]
   [ 0.  1.]]

   [[ 1.  0.]
   [ 0.  1.]]
 ]
]"""