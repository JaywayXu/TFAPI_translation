"""tf.to_double(x, name = 'ToDouble')
解释：这个函数是将一个Tensor的数据类型转换成float64。
使用例子："""

import tensorflow as tf

sess = tf.Session()
data = tf.constant(123)
print (sess.run(data))
d = tf.to_double(data)
print (sess.run(d))
"""输入参数：
  ● x: 一个Tensor或者是SparseTensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor或者SparseTensor，数据类型是float64，数据维度和x相同。
提示：
  ● 错误: 如果x是不能被转换成float64类型的，那么将报错。"""