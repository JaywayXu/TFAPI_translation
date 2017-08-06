"""tf.shape(input, name = None)
解释：这个函数是返回input的数据维度，返回的Tensor数据维度是一维的。
使用例子："""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(sess.run(data))
d = tf.shape(data)
print(sess.run(d))  # [2 2 3]

"""输入参数：
  ● input: 一个Tensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型是int32。"""
