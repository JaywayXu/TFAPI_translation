"""详细计算方法参见tf_sparse_segment_sum"""
import tensorflow as tf

"""tf.sparse_segment_mean(data, indices, segment_ids, name=None)
功能：tensor进行拆分后求平均值。和segment_mean类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。"""

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], tf.float32)
z = tf.sparse_segment_mean(a, tf.constant([0, 1]), tf.constant([0, 0]))
z2 = tf.sparse_segment_mean(a, tf.constant([0, 1]), tf.constant([0, 1]))
z3 = tf.sparse_segment_mean(a, tf.constant([0, 2]), tf.constant([0, 1]))
z4 = tf.sparse_segment_mean(a, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))

sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
print(sess.run(z4))
sess.close()
# z==>[[3. 4. 5. 6.]]
# z2==>[[1. 2. 3. 4.]
#       [5. 6. 7. 8.]]
# z3==>[[1. 2. 3. 4.]
#       [9. 10. 11. 12.]]
# z4==>[[3. 4. 5. 6.]
#       [9. 10. 11. 12.]]
