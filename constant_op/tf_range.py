import tensorflow as tf

"""tf.range(start, limit=None, delta=1, dtype=None, name='range')
功能：生成一个序列值，从start开始，每次递增delta，直到不超过limit的值结束。
输入：start：起始值;
     limit：限制值，不能超过;
     delta：步长。"""

a = tf.range(1, 10, 3)

sess = tf.Session()
print(sess.run(a))
sess.close()
# a==>[1 4 7]
