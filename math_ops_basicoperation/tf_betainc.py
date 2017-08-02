import tensorflow as tf

"""tf.betainc(a,b,x,name=None)
功能：计算I_x(a, b)。I_x(a, b) = {B(x; a, b)}/{B(a, b)}。
                    B(x; a, b) = \intergral_from_0_to_x t^{a-1} (1 - t)^{b-1} dt。
                    B(a, b) = \intergral_from_0_to_1 t^{a-1} (1 - t)^{b-1} dt。即完全beta函数。          
输入：x为张量，可以为`float32`, `float64`类型。a,b与x同类型。"""
a = tf.constant(1, tf.float64)
b = tf.constant(1, tf.float64)
x = tf.constant([[0, 0.5, 1]], tf.float64)
z = tf.betainc(a, b, x)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[0. 0.5 1.]]
