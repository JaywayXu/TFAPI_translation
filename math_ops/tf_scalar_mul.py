"""1.4 tf.scalar_mul(scalar,x)
功能：固定倍率缩放。
输入：scalar必须为0维元素，x为tensor。"""
import tensorflow as tf

scalar = 2.2
x = tf.constant([[1.2, -1.0]], tf.float64)
z = tf.scalar_mul(scalar, x)
sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[2.64,-2.2]]
