"""检查两个张量值是否相等"""
import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([2, 2, 2, 5, 5, 5])
sess = tf.Session()
print(sess.run(tf.equal(a, b)))
"""检查张量值是否不相等"""
import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([2, 2, 2, 5, 5, 5])
sess = tf.Session()
print(sess.run(tf.not_equal(a, b)))
# [False  True False False  True False]
# [ True False  True  True False  True]

a = tf.constant([[1], [2], [3], [1]])  # shape为(4, 1)
b = tf.to_float(tf.equal(a, [1]))  # shape为(1)
with tf.Session() as sess:
    print(sess.run(b))
"""equal函数支持广播模式,例如:
[[ 1.]
 [ 0.]
 [ 0.]
 [ 1.]]"""