import tensorflow as tf

"""tf.norm(tensor, ord='euclidean', axis=None, keep_dims=False, name=None)
功能：求取范数。
输入：ord：范数类型，默认为‘euclidean’，支持的有‘fro’，‘euclidean’，‘0’，‘1’，‘2’，‘np.inf’;
         axis：默认为‘None’，tensor为向量。
     keep_dims:默认为‘None’，结果为向量，若为True，保持维度。"""

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=tf.float32)
z1 = tf.norm(a)
z2 = tf.norm(a, ord=1)
z3 = tf.norm(a, ord=2)
z4 = tf.norm(a, ord=1, axis=0)
z5 = tf.norm(a, ord=1, axis=1)
z6 = tf.norm(a, ord=1, axis=1, keep_dims=True)

sess = tf.Session()
print(sess.run(z1))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
print(sess.run(z5))
print(sess.run(z6))
sess.close()

# z==>9.53939
# z2==>21.0
# z3==>9.53939
# z4==>[5.  7.  9.]
# z5==>[6.  15.]
# z6==>[[6.]
#       [15.]]
