import tensorflow as tf

"""tf.reduce_logsumexp(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算log(sum(exp()))，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[0, 0, 0], [0, 0, 0]], dtype=tf.float64)
z = tf.reduce_logsumexp(a)
z2 = tf.reduce_logsumexp(a, 0)
z3 = tf.reduce_logsumexp(a, 1)

sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
sess.close()

# z==>1.79175946923#log(6)
# z2==>[0.69314718 0.69314718 0.69314718]#[log(2) log(2) log(2)]
# z3==>[1.09861229 1.09861229]#[log(3) log(3)]
