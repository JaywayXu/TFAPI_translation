import tensorflow as tf


"""tf.sparse_segment_sqrt_n(data, indices, segment_ids, name=None)
功能：tensor进行拆分后求和再除以N的平方根。N为reduce segment数量。
     和segment_mean类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。
 N is the size of the segment being reduced."""

a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], tf.float32)
z = tf.sparse_segment_sqrt_n(a, tf.constant([0, 1]), tf.constant([0, 0]))

z2 = tf.sparse_segment_sqrt_n(a, tf.constant([0, 1]), tf.constant([0, 1]))
z3 = tf.sparse_segment_sqrt_n(a, tf.constant([0, 2]), tf.constant([0, 1]))
z4 = tf.sparse_segment_sqrt_n(a, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))

sess = tf.Session()
print(sess.run(z))


sess.close()


# z==>[[4.24264069 5.65685424 7.07106781 8.48528137]]
# 这个结果是将0行和1行相加得到[6,8,10,12]后处以sqrt(2)就是这个奇奇怪怪的结果啦。而这个2表示的是[0,1]中向量的维度
# z2==>[[1. 2. 3. 4.]
#       [5. 6. 7. 8.]]
# z3==>[[1. 2. 3. 4.]
#       [9. 10. 11. 12.]]
# z4==>[[4.24264069 5.65685424 7.07106781 8.48528137]
#       [9. 10. 11. 12.]]
