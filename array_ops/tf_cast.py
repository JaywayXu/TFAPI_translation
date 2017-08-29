"""tf.cast(x, dtype, name = None)
解释：这个函数是将一个Tensor或者SparseTensor的数据类型转换成dtype。
使用例子："""

import tensorflow as tf

sess = tf.Session()
data = tf.constant([x for x in range(20)], tf.float32)
print(sess.run(data))
d = tf.cast(data, tf.int32)
print(sess.run(d))
"""输入参数：
  ● x: 一个Tensor或者是SparseTensor。
  ● dtype: 目标数据类型。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor或者SparseTensor，数据维度和x相同。
提示：
  ● 错误: 如果x是不能被转换成dtype类型的，那么将报错。"""
"""cast函数与条件语句混用
import tensorflow as tf

a = tf.constant([[0.3], [0.5], [0.6], [0.7]])
predicted = tf.cast(a > 0.5, tf.float32)
with tf.Session() as sess:
    print(sess.run(predicted))
# [[ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]]
"""
