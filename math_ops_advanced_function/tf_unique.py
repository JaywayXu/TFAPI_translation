import tensorflow as tf

"""tf.unique(x, out_idx=None, name=None)
功能：罗列非重复元素及其编号。"""

a = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 9, 1])
y, idx = tf.unique(a)

sess = tf.Session()
print(sess.run(y))
print(sess.run(idx))
sess.close()

# y==>[1 2 4 7 8 9]
# idx==>[0 0 1 2 2 2 3 4 5 0]
