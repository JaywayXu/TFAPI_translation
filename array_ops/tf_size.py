"""tf.size(input, name = None)
解释：这个函数是返回input中一共有多少个元素。
使用例子："""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(sess.run(data))
d = tf.size(data)
print(sess.run(d))  # 12
"""输入参数：
  ● input: 一个Tensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型是int32。"""
