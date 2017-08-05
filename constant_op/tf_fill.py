import tensorflow as tf

"""tf.fill(dims, value, name=None)
功能：生成一个值全为value的tensor，其形状与dims相同。
输入：dims：一维的int32型的列表，"""

a = tf.fill([2, 3], 7)

sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[7. 7. 7.]
#      [7. 7. 7.]]
