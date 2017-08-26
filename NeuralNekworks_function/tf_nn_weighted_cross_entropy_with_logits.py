"""weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None):
此函数功能以及计算方式基本与tf_nn_sigmoid_cross_entropy_with_logits差不多,但是加上了权重的功能,是计算具有权重的sigmoid交叉熵函数
计算方法 :pos_weight*targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))
官方文档定义及推导过程:
通常的cross-entropy交叉熵函数定义如下:
      targets * -log(sigmoid(logits)) +
          (1 - targets) * -log(1 - sigmoid(logits))

对于加了权值pos_weight的交叉熵函数:

      targets * -log(sigmoid(logits)) * pos_weight +
          (1 - targets) * -log(1 - sigmoid(logits))

现在我们使用 `x = logits`, `z = targets`, `q = pos_weight`的代数式
  The loss is:

        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

  我们把`l = (1 + (q - 1) * z)`, 来确保稳定性并且比避免溢出,公式为:

      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

  `logits` and `targets` 必须要有相同的数据类型和shape.
参数:
_sentinel:本质上是不用的参数，不用填

targets:一个和logits具有相同的数据类型（type）和尺寸形状（shape）的张量（tensor）

shape:[batch_size,num_classes],单样本是[num_classes]

logits:一个数据类型（type）是float32或float64的张量

pos_weight:正样本的一个系数

name:操作的名字，可填可不填
"""
import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(3, 3), dtype=tf.float32)
# np.random.rand()传入一个shape,返回一个在[0,1)区间符合均匀分布的array

output = tf.nn.weighted_cross_entropy_with_logits(logits=input_data,
                                                  targets=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                                                  pos_weight=2.0)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
# [[ 1.04947078  0.89594436  0.92146152]
#  [ 0.70252579  1.00673866  1.08856964]
#  [ 1.07195592  1.18525708  1.04106498]]
