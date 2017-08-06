"""tf.squeeze(input, squeeze_dims = None, name = None)
解释：这个函数的作用是将input中维度是1的那一维去掉。但是如果你不想把维度是1的全部去掉，
那么你可以使用squeeze_dims参数，来指定需要去掉的位置。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[1, 2, 1], [3, 1, 1]])
print(sess.run(tf.shape(data)))
d_1 = tf.expand_dims(data, 0)
d_1 = tf.expand_dims(d_1, 2)
d_1 = tf.expand_dims(d_1, -1)
d_1 = tf.expand_dims(d_1, -1)
print(sess.run(tf.shape(d_1)))
# [1, 2, 1, 3, 1, 1]
d_2 = d_1
print(sess.run(tf.shape(tf.squeeze(d_1))))
# shape(squeeze(t)) ==> [2, 3]
print(sess.run(tf.shape(tf.squeeze(d_2, [2, 4]))))
# shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]

"""输入参数：
  ● input: 一个Tensor。
  ● squeeze_dims: （可选）一个序列，索引从0开始，只移除该列表中对应位的tensor。默认下，是一个空序列[]。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和输入数据相同。"""
