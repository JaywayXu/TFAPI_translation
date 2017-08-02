import tensorflow as tf

"""tf.igammac(a,x,name=None)
功能：计算gamma(a,x)/gamma(a),gamma(a,x)=\intergral_from_x_to_inf t^(a-1)*exp^(-t)dt。
输入：x为张量，可以为`float32`, `float64`类型。"""

a = tf.constant(1, tf.float64)
x = tf.constant([[-1, 0, 1, 2, 3]], tf.float64)
z = tf.igammac(a, x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[        nan  1.          0.36787944  0.13533528  0.04978707]]
