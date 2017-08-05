import tensorflow as tf

"""tf.ones_like(tensor, dtype=None, name=None, optimize=True)
功能：生成一个值全为1的tensor，其形状与输入tensor相同。
输入：dtype:未指定时返回tesnsor的类型"""

x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
a = tf.ones_like(x)

sess = tf.Session()
print(sess.run(a))
sess.close()
# a==>[[1. 1. 1.]
#      [1. 1. 1.]]
