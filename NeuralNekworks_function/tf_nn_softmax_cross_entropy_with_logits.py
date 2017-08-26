"""def softmax_cross_entropy_with_logits(_sentinel=None,  # pylint: disable=invalid-name
                                      labels=None, logits=None,
                                      dim=-1, name=None)
解释：这个函数的作用是计算 logits 经 softmax 函数激活之后的交叉熵。
对于每个独立的分类任务，这个函数是去度量概率误差。比如，在 CIFAR-10 数据集上面，每张图片只有唯一一个分类标签：一张图可能是一只狗或者一辆卡车，
但绝对不可能两者都在一张图中。（这也是和 tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)这个API的区别）
警告：输入API的数据 logits 不能进行缩放，因为在这个API的执行中会进行 softmax 计算，如果 logits 进行了缩放，那么会影响计算正确率。
不要调用这个API去计算 softmax 的值，因为这个API最终输出的结果并不是经过 softmax 函数的值。
logits 和 labels 必须有相同的数据维度 [batch_size, num_classes]，和相同的数据类型 float32 或者 float64 。
它适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象."""

import tensorflow as tf

input_data = tf.Variable([[0.2, 0.1, 0.9], [0.3, 0.4, 0.6]], dtype=tf.float32)
output = tf.nn.softmax_cross_entropy_with_logits(logits=input_data, labels=[[0, 0, 1], [1, 0, 0]])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
    # [1.36573195]
"""
输入参数：
    _sentinel: 这个参数一般情况不使用,直接设置为None就好
    logits: 一个没有缩放的对数张量。labels和logits具有相同的数据类型（type）和尺寸（shape）
    labels: 每一行 labels[i] 必须是一个有效的概率分布值。
    name: 为这个操作取个名字。
输出参数：
一个 Tensor ，数据维度是一维的，长度是 batch_size，数据类型都和 logits 相同。
"""
