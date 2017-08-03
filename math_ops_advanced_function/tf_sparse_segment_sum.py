import tensorflow as tf

"""tf.sparse_segment_sum(data, indices, segment_ids, name=None)
功能：tensor进行拆分后求和。和segment_sum类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。"""

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
z = tf.sparse_segment_sum(a, tf.constant([0, 1]), tf.constant([0, 0]))
# 选择前两行，并且计算前两行的和
z2 = tf.sparse_segment_sum(a, tf.constant([0, 1]), tf.constant([0, 1]))
# 选择前两行，但是只计算第0行的和，第一行保留原样
z3 = tf.sparse_segment_sum(a, tf.constant([0, 2]), tf.constant([0, 1]))
# 选择第0行和第2行但是只计算第0行，第2行保留原样
z4 = tf.sparse_segment_sum(a, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
# 选择三行，但是只是计算0行和1行，第2行保留原样
sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()

# z==>[[6 8 10 12]]
# z2==>[[1 2 3 4]
#       [5 6 7 8]]
# z3==>[[1 2 3 4]
#       [9 10 11 12]]
# z4==>[[6 8 10 12]
#       [9 10 11 12]]
