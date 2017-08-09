"""tf.nn.bias_add(value, bias, name = None)
解释：这个函数的作用是将偏差项 bias 加到 value 上面。
这个操作你可以看做是 tf.add 的一个特例，其中 bias 必须是一维的。该API支持广播形式，因此 value 可以有任何维度。
但是，该API又不像 tf.add ，可以让 bias 的维度和 value 的最后一维不同。具体看使用例子。"""

import tensorflow as tf

a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
b = tf.constant([2.0, 1.0])
c = tf.constant([1.0])
sess = tf.Session()
print(sess.run(tf.nn.bias_add(a, b)))
# 因为 a 最后一维的维度是 2 ，但是 c 的维度是 1，所以以下语句将发生错误
# print (sess.run(tf.nn.bias_add(a, c)))
# ValueError: Dimensions must be equal, but are 2 and 1 for 'BiasAdd_1' (op: 'BiasAdd') with input shapes: [3,2], [1].
# 但是 tf.add() 可以正确运行
print(sess.run(tf.add(a, c)))
"""输入参数：
  ● value: 一个Tensor。数据类型必须是float，double，int64，int32，uint8，int16，int8或者complex64。
  ● bias: 一个一维的Tensor，数据维度和 value 的最后一维相同。数据类型必须和value相同。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和value相同。"""
