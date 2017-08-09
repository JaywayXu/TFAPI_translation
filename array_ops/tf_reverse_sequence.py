"""tf.reverse_sequence(input, seq_lengths, seq_dim, name = None)
解释：将input中的值沿着第seq_dim维度进行翻转。
这个操作先将input沿着第0维度切分，然后对于每个切片，将切片长度为seq_lengths[i]的值，沿着第seq_dim维度进行翻转。
向量seq_lengths中的值必须满足seq_lengths[i] < input.dims[seq_dim]，并且其长度必须是input_dims(0)。
对于每个切片i的输出，我们将第seq_dim维度的前seq_lengths[i]的数据进行翻转。"""
# 比如：
# # Given this:
# seq_dim = 1
# input.dims = (4, 10, ...)
# seq_lengths = [7, 2, 3, 5]
#
# # 因为input的第0维度是4，所以先将input切分成4个切片；
# # 因为seq_dim是1，所以我们按着第1维度进行翻转。
# # 因为seq_lengths[0] = 7，所以我们第一个切片只翻转前7个值，该切片的后面的值保持不变。
# # 因为seq_lengths[1] = 2，所以我们第一个切片只翻转前2个值，该切片的后面的值保持不变。
# # 因为seq_lengths[2] = 3，所以我们第一个切片只翻转前3个值，该切片的后面的值保持不变。
# # 因为seq_lengths[3] = 5，所以我们第一个切片只翻转前5个值，该切片的后面的值保持不变。
# output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
# output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
# output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
# output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
#
# output[0, 7:, :, ...] = input[0, 7:, :, ...]
# output[1, 2:, :, ...] = input[1, 2:, :, ...]
# output[2, 3:, :, ...] = input[2, 3:, :, ...]
# output[3, 2:, :, ...] = input[3, 2:, :, ...]
# 使用例子：

import tensorflow as tf
# tf.reverse_sequence(input, seq_lengths, seq_dim, name = None)
sess = tf.Session()
input = tf.constant([[1, 2, 3, 4], [3, 4, 5, 6]], tf.int64)
seq_lengths = tf.constant([3, 2], tf.int64)
seq_dim = 1
output = tf.reverse_sequence(input, seq_lengths, seq_dim)
print(sess.run(output))
sess.close()
# 原本是[1, 2, 3, 4]
#      [3, 4, 5, 6]
#  将第一行中的前三个数进行位置的反转，对于第二行的前两个数进行位置的反转
# output
# [[3 2 1 4]
#  [4 3 5 6]]
"""输入参数：
  ● input: 一个Tensor，需要反转的数据。
  ● seq_lengths: 一个Tensor，数据类型是int64，数据长度是input.dims(0)，并且max(seq_lengths) < input.dims(seq_dim)。
  ● seq_dim: 一个int，确定需要翻转的维度。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和input相同，数据维度和input相同。"""