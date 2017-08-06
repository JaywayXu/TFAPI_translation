"""tf.rank(input, name = None)
解释：这个函数是返回Tensor的秩。
注意：Tensor的秩和矩阵的秩是不一样的，Tensor的秩指的是元素维度索引的数目，
这个概念也被成为order, degree或者ndims。比如，一个Tensor的维度是[1, 28, 28, 1]，那么它的秩就是4。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(sess.run(data))
d = tf.rank(data)
print(sess.run(tf.shape(data)))  # [2 2 3]
print(sess.run(d))  # 3
"""输入参数：
  ● input: 一个Tensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型是int32。"""
