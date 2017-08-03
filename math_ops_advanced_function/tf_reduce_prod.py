import tensorflow as tf

"""tf.reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
功能：沿着维度axis计算元素积，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。"""

a = tf.constant([[1, 2, 3], [4, 5, 6]])
z = tf.reduce_prod(a)
z2 = tf.reduce_prod(a, 0)
z3 = tf.reduce_prod(a, 1)
z4 = tf.reduce_prod(a, 1, keep_dims=True)
sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()

# z==>720
# z2==>[4 10 18]
# z3==>[6 120]
# z4==>[[  6]
#      [120]]
