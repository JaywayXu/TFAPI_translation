"""逐元素计算x>y的真值表"""
import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([2, 2, 2, 5, 5, 5])
sess = tf.Session()
print(sess.run(tf.greater(a, b)))
# [False False  True False False  True]
"""逐元素计算x>=y的真值表"""
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([2, 2, 2, 5, 5, 5])
sess = tf.Session()
print(sess.run(tf.greater_equal(a, b)))
# [False  True  True False  True  True]
