import tensorflow as tf

"""tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
功能：从正太分布中随机输出。
输入：shape：一维整型tensor，指定tensor的形状;
     mean：正太分布的平均值，默认为0;
     stddev：正太分布的标准差，默认为1;
     seed：随机种子。
     随机种子（Random Seed）是计算机专业术语，
     一种以随机数作为对象的以真随机数（种子）为初始条件的随机数。
     一般计算机的随机数都是伪随机数，以一个真随机数（种子）作为初始条件，
     然后用一定的算法不停迭代产生随机数"""

a = tf.random_normal([2, 2], seed=112)

sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[-0.72891599 -1.35909426]
#      [ 0.06045228  1.12680387]]
