"""tf.expand_dims(input, dim, name = None)
解释：这个函数的作用是向input中插入维度是1的张量。
我们可以指定插入的位置dim，dim的索引从0开始，dim的值也可以是负数，从尾部开始插入，符合 python 的语法。
这个操作是非常有用的。举个例子，如果你有一张图片，数据维度是[height, width, channels]，你想要加入“批量”这个信息，
那么你可以这样操作expand_dims(images, 0)，那么该图片的维度就变成了[1, height, width, channels]。
这个操作要求：
-1-input.dims() <= dim <= input.dims()
这个操作是squeeze()函数的相反操作，可以一起灵活运用。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[1, 2, 1], [3, 1, 1]])
print(sess.run(tf.shape(data)))
# [2 3]
d_1 = tf.expand_dims(data, 0)
print(sess.run(tf.shape(d_1)))
# [1 2 3]
print("1:add dimension to 0 ", d_1)
# 1:add dimension to 0  Tensor("ExpandDims:0", shape=(1, 2, 3), dtype=int32)
d_1 = tf.expand_dims(d_1, 2)
print(sess.run(tf.shape(d_1)))
# [1 2 1 3]
print("2:add dimension to 2 ", d_1)
# 2:add dimension to 2  Tensor("ExpandDims_1:0", shape=(1, 2, 1, 3), dtype=int32)
d_1 = tf.expand_dims(d_1, -1)
print(sess.run(tf.shape(d_1)))
# [1 2 1 3 1]
print("3:add dimension to -1 ", d_1)
# 3:add dimension to -1  Tensor("ExpandDims_2:0", shape=(1, 2, 1, 3, 1), dtype=int32)
"""输入参数：
  ● input: 一个Tensor。
  ● dim: 一个Tensor，数据类型是int32，标量。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和输入数据相同，数据和input相同，但是维度增加了一维。"""



