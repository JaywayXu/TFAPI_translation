import tensorflow as tf

"""tf.polygamma(a,x,name=None)
功能：计算psi^{(a)}(x),psi^{(a)}(x) = ({d^a}/{dx^a})*psi(x),psi即为polygamma。    
输入：x为张量，可以为`float32`, `float64`类型。a=tf.constant(1,tf.float64) """

a = tf.constant(1, tf.float64)
x = tf.constant([[1, 2, 3, 4]], tf.float64)
z = tf.polygamma(a, x)

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1.64493407 0.64493407 0.39493407 0.28382296]]
