"""tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
功能：从截断正太分布中随机输出。
输入：shape：一维整型tensor，指定tensor的形状;
     mean：正太分布的平均值，默认为0;
     stddev：正太分布的标准差，默认为1;
     seed：随机种子。"""
import tensorflow as tf

a = tf.truncated_normal([2, 2], seed=112)
sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[-0.72891599 -1.35909426]
#      [ 0.06045228  1.12680387]]
