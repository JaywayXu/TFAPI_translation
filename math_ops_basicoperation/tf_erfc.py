import tensorflow as tf

"""tf.erfc(x,name=None)
功能：计算x高斯互补误差。
输入：x为张量，可以为`half`,`float32`, `float64`类型。"""

x = tf.constant([[-1, 0, 1, 2, 3]], tf.float64)
z = tf.erfc(x)

sess = tf.Session()
print(sess.run(z))
sess.close()

# [[  1.84270079e+00   1.00000000e+00   1.57299207e-01   4.67773498e-03 2.20904970e-05]]