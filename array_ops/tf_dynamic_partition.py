"""tf.dynamic_partition(data, partitions, num_partitions, name = None)动态分区
解释：根据从partitions中取得的索引，将data分割成num_partitions份。
我们先从partitions.ndim 中取出一个元祖js，那么切片data[js, ...]将成为输出数据outputs[partitions[js]]的一部分。
我们将js按照字典序排列，即js里面的值为(0, 0, ..., 1, 1, ..., 2, 2, ..., ..., num_partitions - 1, num_partitions - 1, ...)。
我们将partitions[js] = i的值放入outputs[i]。outputs[i]中的第一维对应于partitions.values == i的位置。
"""

"""# Scalar partitions
partitions = 1
num_partitions = 2
data = [10, 20]
outputs[0] = []  # Empty with shape [0, 2]
outputs[1] = [[10, 20]]

# Vector partitions 向量分区
partitions = [0, 0, 1, 1, 0]  取得索引
num_partitions = 2 将数据分割成两份
data = [10, 20, 30, 40, 50]
outputs[0] = [10, 20, 50]
outputs[1] = [30, 40]
"""

import tensorflow as tf

sess = tf.Session()
num_partitions = 2
partitions = [0, 0, 1, 1, 0]
data = [10, 20, 30, 40, 50]
output = tf.dynamic_partition(data, partitions=partitions, num_partitions=num_partitions)
print(sess.run(output))
# [array([10, 20, 50]), array([30, 40])]
sess.close()
"""输入参数：
  ● data: 一个Tensor。
  ● partitions: 一个Tensor，数据类型必须是int32。任意数据维度，但其中的值必须是在范围[0, num_partitions)。
  ● num_partitions: 一个int，其值必须不小于1。输出的切片个数。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个数组Tensor，数据类型和data相同。"""
