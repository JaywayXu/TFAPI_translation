"""tf.ones(shape, dtype=tf.float32, name=None)
功能：生成一个值全为1的tensor。默认为float32类型。
输入：shape：一维的int32型的列表。"""
import tensorflow as tf

a = tf.ones([2, 3])
sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[1. 1. 1.]
#      [1. 1. 1.]]
