"""tf.nn.softmax(logits, name=None)
解释：这个函数的作用是计算 softmax 激活函数。
对于每个批 i 和 分类 j，我们可以得到：
softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
使用柔性最大值(softmax)作为激活函数的好处在于其激活值的和为1,并且当一个类别的几率变高时其余的几率就会变低"""

import tensorflow as tf
import math as m

input_data = tf.Variable([[0.2, 0.1, 0.9]], dtype=tf.float32)
output = tf.nn.softmax(input_data)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input_data))
    print(sess.run(output))
    print(sess.run(tf.shape(output)))
    print(m.exp(0.2)/(m.exp(0.2)+m.exp(0.1)+m.exp(0.9)))
"""输入参数：
  ● logits: 一个Tensor。数据类型是以下之一：float32或者float64。数据维度是二维 [batch_size, num_classes]。
  ● name: 为这个操作取个名字。
输出参数：
一个 Tensor ，数据维度和数据类型都和 logits 相同。"""
# [[ 0.2         0.1         0.89999998]]
# [[ 0.25519383  0.23090893  0.51389724]]=
# [[exp(0.2)/(exp(0.2)+exp(0.1)+exp(0.9))],[exp(0.1)/(exp(0.2)+exp(0.1)+exp(0.9))],[exp(0.9)/(exp(0.2)+exp(0.1)+exp(0.9))]]
# [1 3]
