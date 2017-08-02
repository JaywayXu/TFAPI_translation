import tensorflow as tf

"""功能：计算gamma(a,x)/gamma(a),gamma(a,x)=\intergral_from_0_to_x t^(a-1)*exp^(-t)dt。
输入：x为张量，可以为`float32`, `float64`类型。"""

a = tf.constant(1, tf.float64)
x = tf.constant([[1, 2, 3, 4]], tf.float64)
z = tf.igamma(a, x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0.63212056 0.86466472 0.95021293 0.98168436]]
