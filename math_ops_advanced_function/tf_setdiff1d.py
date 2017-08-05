import tensorflow as tf

"""tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)
功能：返回在x里不在y里的元素值和下标，"""

a = tf.constant([1, 2, 3, 4])
b = tf.constant([1, 4])
out, idx = tf.setdiff1d(a, b)


sess = tf.Session()
print(sess.run(out))
print(sess.run(idx))
sess.close()

# out==>[2 3]
# idx==>[1 2]
