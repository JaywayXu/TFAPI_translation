"""tf.transpose(a, perm = None, name = 'transpose')
解释：将a进行转置，并且根据perm参数重新排列输出维度。
输出数据tensor的第i维将根据perm[i]指定。比如，如果perm没有给定，那么默认是perm = [n-1, n-2, ..., 0]，
其中rank(a) = n。默认情况下，对于二维输入数据，其实就是常规的矩阵转置操作。"""
"""
input_data.dims = (1, 4, 3)
perm = [1, 2, 0]

# 因为 output_data.dims[0] = input_data.dims[ perm[0] ]
# 因为 output_data.dims[1] = input_data.dims[ perm[1] ]
# 因为 output_data.dims[2] = input_data.dims[ perm[2] ]
# 所以得到 output_data.dims = (4, 3, 1)
output_data.dims = (4, 3, 1)
"""
import tensorflow as tf

sess = tf.Session()
input_data = tf.constant([[1, 2, 3], [4, 5, 6]])
print(sess.run(tf.transpose(input_data)))
print(sess.run(input_data))
print(sess.run(tf.transpose(input_data, perm=[1, 0])))
input_data = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
print('input_data shape: ', sess.run(tf.shape(input_data)))
# [1, 4, 3]
output_data = tf.transpose(input_data, perm=[1, 2, 0])
# [4, 3, 1]
print('output_data shape: ', sess.run(tf.shape(output_data)))
print(sess.run(output_data))
# [[[ 1]
#   [ 2]
#   [ 3]]
#  [[ 4]
#   [ 5]
#   [ 6]]
#
#  [[ 7]
#   [ 8]
#   [ 9]]
#
#  [[10]
#   [11]
#   [12]]]
"""形式为：[[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]"""
sess.close()
"""输入参数：
  ● a: 一个Tensor。
  ● perm: 一个对于a的维度的重排列组合。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个经过翻转的Tensor。"""
