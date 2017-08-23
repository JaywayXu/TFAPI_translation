import tensorflow as tf

"""逐元素的计算x<y的真值表"""
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([3, 3, 3, 6, 6, 6])
sess = tf.Session()
print(sess.run(tf.less(a, b)))
# [ True  True False  True  True False]

"""主元素的计算x<=y的真值表"""
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([3, 3, 3, 6, 6, 6])
sess = tf.Session()
print(sess.run(tf.less_equal(a, b)))
# [ True  True  True  True  True  True]
