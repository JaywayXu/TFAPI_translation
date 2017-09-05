"""def sigmoid_cross_entropy_with_logits(_sentinel=None,  # pylint: disable=invalid-name
                                      labels=None, logits=None,
                                      name=None):
解释：这个函数的作用是计算经sigmoid 函数激活之后的交叉熵。
为了描述简洁，我们规定 x = logits，z = targets，那么 Logistic 损失值为：
x - x * z + log( 1 + exp(-x) )
对于x<0的情况,为了执行的稳定,使用计算式:
-x * z + log(1 + exp(x))
为了确保计算稳定，避免溢出，真实的计算实现如下：
max(x, 0) - x * z + log(1 + exp(-abs(x)) )
logits 和 targets 必须有相同的数据类型和数据维度。
它适用于每个类别相互独立但互不排斥的情况,在一张图片中，同时包含多个分类目标（大象和狗），那么就可以使用这个函数。
以上是经过化简过后的式子,具体化简过程请参考下方英文文档"""

import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(1, 3), dtype=tf.float32)
# np.random.rand()传入一个shape,返回一个在[0,1)区间符合均匀分布的array

output = tf.nn.sigmoid_cross_entropy_with_logits(logits=input_data, labels=[[1.0, 0.0, 0.0]])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
    # [[ 0.5583781   1.06925142  1.08170223]]
"""
输入参数：
  _sentinel: 一般情况下不怎么使用的参数,可以直接保持默认使其为None
  logits: 一个Tensor。数据类型是以下之一：float32或者float64。
  targets: 一个Tensor。数据类型和数据维度都和 logits 相同。
  name: 为这个操作取个名字。
输出参数：
一个 Tensor ，数据维度和 logits 相同。"""
"""推导过程`x = logits`, `z = labels`.  logistic loss 计算式为:
其中交叉熵(cross entripy)基本函数式
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  对于x<0时,为了避免计算exp(-x)时溢出,我们使用以下这种形式表示

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))
  综合x>0和x<0的情况,我们使用以下函数式
      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `labels` must have the same type and shape.
  注意logits和labels必须具有相同的type和shape"""

