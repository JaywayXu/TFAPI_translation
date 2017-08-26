"""sparse_softmax_cross_entropy_with_logits(_sentinel=None,  # pylint: disable=invalid-name
                                             labels=None, logits=None,
                                             name=None):
此函数大致与tf_nn_softmax_cross_entropy_with_logits的计算方式相同,
适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象
但是在对于labels的处理上有不同之处,labels从shape来说此函数要求shape为[batch_size],
labels[i]是[0,num_classes)的一个索引, type为int32或int64,即labels限定了是一个一阶tensor,并且取值范围只能在分类数之内,表示一个对象只能属于一个类别
_sentinel:本质上是不用的参数，不用填
logits：shape为[batch_size,num_classes],type为float32或float64
name:操作的名字，可填可不填
"""
import tensorflow as tf

input_data = tf.Variable([[0.2, 0.1, 0.9], [0.3, 0.4, 0.6]], dtype=tf.float32)
output = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input_data, labels=[0, 2])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
# [ 1.36573195  0.93983102]