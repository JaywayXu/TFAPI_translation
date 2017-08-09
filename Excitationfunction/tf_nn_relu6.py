"""tf.nn.relu6(features, name = None)
解释：这个函数的作用是计算激活函数relu6，即min(max(features, 0), 6)。
相比于原来的函数relu如果最大值大于6的话会归一化到6
使用例子："""
import tensorflow as tf

a = tf.constant([-1.0, 12.0])
with tf.Session() as sess:
    b = tf.nn.relu6(a)
    print(sess.run(b))
"""输入参数：
  ● features: 一个Tensor。数据类型必须是：float，double，int32，int64，uint8，int16或者int8。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和features相同。"""
