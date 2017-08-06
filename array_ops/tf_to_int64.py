"""tf.to_int64(x, name = 'ToInt64')
解释：这个函数是将一个Tensor的数据类型转换成int64。
使用例子："""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([x for x in range(20)], tf.float32)
print(sess.run(data))
d = tf.to_int64(data)
print(sess.run(d))
"""输入参数：
  ● x: 一个Tensor或者是SparseTensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor或者SparseTensor，数据类型是int64，数据维度和x相同。
提示：
  ● 错误: 如果x是不能被转换成int64类型的，那么将报错。"""
