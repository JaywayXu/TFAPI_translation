import tensorflow as tf

"""tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
功能：从均匀分布中随机输出，默认区域为‘[0,1)’。
输入：shape：一维整型tensor，指定tensor的形状;
     minval：均匀分布的最小值，默认为0;
     maxval：均匀分布的最大值，如果类型为float32则默认为1;
     seed：随机种子。"""

a = tf.random_uniform([2, 2], seed=112)

sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[0.30445623 0.57834935]
#      [0.52905083 0.00853038]]
