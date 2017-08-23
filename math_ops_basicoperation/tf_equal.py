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
