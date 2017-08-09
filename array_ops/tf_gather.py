"""tf.gather(params, indices, name = None)
解释：根据indices索引，从params中取对应索引的值，然后返回。
indices必须是一个整型的tensor，数据维度是常量或者一维。
最后输出的数据维度是indices.shape + params.shape[1:]。"""
"""# Scalar indices
output[:, ..., :] = params[indices, :, ... :]

# Vector indices
output[i, :, ..., :] = params[indices[i], :, ... :]

# Higher rank indices
output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
如果indices是一个从0到params.shape[0]的排列，即len(indices) = params.shape[0]，
那么这个操作将把params进行重排列。"""

import tensorflow as tf

sess = tf.Session()
params = tf.constant([6, 3, 4, 1, 5, 9, 10])
indices = tf.constant([2, 0, 2, 5])
output = tf.gather(params, indices)
print(sess.run(output))
# [4 6 4 9]
sess.close()
"""输入参数：
  ● params: 一个Tensor。
  ● indices: 一个Tensor，数据类型必须是int32或者int64。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和params相同。"""
