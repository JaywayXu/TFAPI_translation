"""tf.dynamic_stitch(indices, data, name = None)
解释：这是一个交错合并的操作，我们根据indices中的值，将data交错合并，并且返回一个合并之后的tensor。
如下构建一个合并的tensor：
merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
其中，m是一个从0开始的索引。如果indices[m]是一个标量或者向量，那么我们可以得到更加具体的如下推导：
# Scalar indices
merged[indices[m], ...] = data[m][...]

# Vector indices
merged[indices[m][i], ...] = data[m][i, ...]
从上式的推导，我们也可以看出最终合并的数据是按照索引从小到大排序的。那么会产生两个问题：
1）假设如果一个索引同时存在indices[m][i]和indices[n][j]中，其中(m, i) < (n, j)。那么，data[n][j]将作为最后被合并的值。
2）假设索引越界了，那么缺失的位上面的值将被随机值给填补。
比如：
indices[0] = 6
indices[1] = [4, 1]
indices[2] = [[5, 2], [0, 3]]
data[0] = [61, 62]  意思是[61,62]这个数据将放到第六个位置
data[1] = [[41, 42], [11, 12]]  [41,42]将放到第四个位置，[11,12]放到第一个位置
data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]  indices[2]的数据形式和data[2]中的形式完全一致，表示了数据的存放位置
merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
          [51, 52], [61, 62]]"""

import tensorflow as tf

sess = tf.Session()
indices = [6, [4, 1], [[5, 2], [0, 3]]]
data = [[61, 62], [[41, 42], [11, 12]], [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]]
output = tf.dynamic_stitch(indices, data)
print(sess.run(output))
# [[ 1  2]
#  [11 12]
#  [21 22]
#  [31 32]
#  [41 42]
#  [51 52]
#  [61 62]]
# 缺少了第6，第7的位置，索引最后合并的数据中，这两个位置的值会被用随机数代替
indices = [8, [4, 1], [[5, 2], [0, 3]]]
output = tf.dynamic_stitch(indices, data)
# 第一个2被覆盖了，最后合并的数据是第二个2所指的位置
indices = [6, [4, 1], [[5, 2], [2, 3]]]
output = tf.dynamic_stitch(indices, data)
print(sess.run(output))
#  [[-2058359536         198]
#  [         11          12]
#  [          1           2]
#  [         31          32]
#  [         41          42]
#  [         51          52]
#  [         61          62]]
print(sess.run(output))
# [[-2058359536         198]
#  [         11          12]
#  [          1           2]
#  [         31          32]
#  [         41          42]
#  [         51          52]
#  [         61          62]]
sess.close()
"""输入参数：
  ● indices: 一个列表，至少包含两Tensor，数据类型是int32。
  ● data: 一个列表，里面Tensor的个数和indices相同，并且拥有相同的数据类型。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和data相同。"""
