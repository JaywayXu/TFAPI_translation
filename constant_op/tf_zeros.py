import tensorflow as tf

"""tf.zeros(shape, dtype=tf.float32, name=None)
功能：生成一个值全为0的tensor。默认为float32类型。
输入：shape：一维的int32型的列表。"""

a = tf.zeros([2, 3])
b = tf.zeros([3, 2, 2, 2])
sess = tf.Session()
print(sess.run(a))
print(sess.run(b))
sess.close()

# a==>[[0. 0. 0.]
#      [0. 0. 0.]]
# # b==>[[[[ 0.  0.]
#    [ 0.  0.]]
#
#   [[ 0.  0.]
#    [ 0.  0.]]]
#
#
#  [[[ 0.  0.]
#    [ 0.  0.]]
#
#   [[ 0.  0.]
#    [ 0.  0.]]]
#
#
#  [[[ 0.  0.]
#    [ 0.  0.]]
#
#   [[ 0.  0.]
#    [ 0.  0.]]]]

