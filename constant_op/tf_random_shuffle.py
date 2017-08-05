import tensorflow as tf

"""tf.random_shuffle(value, seed=None, name=None)
功能：将tensor第一个维度的数据重新随机排列。
输入：value：tensor。
    seed：随机种子。"""
x = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
a = tf.random_shuffle(x, seed=11)

sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[5 6]
#      [3 4]
#      [1 2]]
