"""tf.sigmoid(x, name = None)
解释：这个函数的作用是计算 x 的 sigmoid 函数。具体计算公式为 y = 1 / (1 + exp(-x))。
函数的返回值位于区间[0.0 , 1.0]中,当输入值较大时,tf.sigmoid将返回一个接近于1.0的值,而当输入值较小时,返回值将接近于0.0.
对于在真实输出位于[0.0,1.0]的样本上训练的神经网络,sigmoid函数可将输出保持在[0.0,1.0]内的能力非常有用.
当输出接近于饱和或者剧烈变化是,对输出返回的这种缩减会带来一些不利影响.
当输入为0时,sigmoid函数的输出为0.5,即sigmoid函数值域的中间点
使用例子："""

import tensorflow as tf

a = tf.constant([[-1.0, -2.0], [1.0, 2.0], [0.0, 0.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))

# [[ 0.26894143  0.11920292]
#  [ 0.7310586   0.88079703]
#  [ 0.5         0.5       ]]
"""
输入参数：
  ● x: 一个Tensor。数据类型必须是float，double，int32，complex64，int64或者qint32。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，如果 x.dtype != qint32 ，那么返回的数据类型和x相同，否则返回的数据类型是 quint8 。"""
