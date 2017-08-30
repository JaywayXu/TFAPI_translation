import tensorflow as tf

"""tf.argmax(input, axis=None, name=None, dimension=None)
功能：返回沿axis维度最大值的下标。
利用Ctrl+B我们可以观察到,argmax函数返回的是arg_max函数的值
也就是说其内层封装的是arg_max函数"""

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], tf.float64)
z1 = tf.argmax(a, axis=0)
z2 = tf.argmax(a, axis=1)

sess = tf.Session()
print(sess.run(z1))
print(sess.run(z2))

sess.close()

# z1==>[2 2 2 2]
# z2==>[3 3 3]
# a.shape=(3, 4)
# 这个操作可以看做是一个降维的操作.
# 注意返回的是最大值的下标而不是最大值
# 在0维度上操作后shape为(4)在1维度上操作后shape为(3)原来的那一维被消去了.
