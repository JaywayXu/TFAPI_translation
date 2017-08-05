import tensorflow as tf

"""tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
功能：生成一个常量tensor
输入：value：一个常量，或者一个list;
     dtype：数据类型;
     shape：生成形状。"""

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant(2, shape=[2, 3])

sess = tf.Session()
print(sess.run(a))
print(sess.run(b))
sess.close()
# a==>[[1 2 3]
#      [4 5 6]]
# b==>[[2 2 2]
#      [2 2 2]]
