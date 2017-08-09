"""tf.nn.dropout(x, keep_prob, noise_shape = None, seed = None, name = None)
解释：这个函数的作用是计算神经网络层的dropout。
一个神经元将以概率keep_prob决定是否放电，如果不放电，那么该神经元的输出将是0，
如果该神经元放电，那么该神经元的输出值将被放大到原来的1/keep_prob倍。
这里的放大操作是为了保持神经元输出总个数不变。比如，神经元的值为[1, 2]，keep_prob的值是0.5，
并且是第一个神经元是放电的，第二个神经元不放电，那么神经元输出的结果是[2, 0]，也就是相当于，
第一个神经元被当做了1/keep_prob个输出，即2个。这样保证了总和2个神经元保持不变。
默认情况下，每个神经元是否放电是相互独立的。但是，如果noise_shape被修改了，
那么他对于变量x就是一个广播形式，而且当且仅当 noise_shape[i] == shape(x)[i] ，
x中的元素是相互独立的。比如，如果 shape(x) = [k, l, m, n], noise_shape = [k, 1, 1, n] ，
那么每个批和通道都是相互独立的，但是每行和每列的数据都是关联的，即要不都为0，要不都还是原来的值。"""

import tensorflow as tf
# tf.nn.dropout(x, keep_prob, noise_shape = None, seed = None, name = None)
a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.Session() as sess:
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 4])  # 第0维相互独立，第1维相互独立
    print(sess.run(b))
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 1])  # 第0维相互独立，第1维不是相互独立的
    print(sess.run(b))
"""输入参数：
  ● x: 一个Tensor。
  ● keep_prob: 一个 Python 的 float 类型。表示元素是否放电的概率。
  ● noise_shape: 一个一维的Tensor，数据类型是int32。代表元素是否独立的标志。
  ● seed: 一个Python的整数类型。设置随机种子。
  ● name: （可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据维度和x相同。
异常：
  ● 输入异常: 如果 keep_prob 不是在 (0, 1]区间，那么会提示错误。"""
