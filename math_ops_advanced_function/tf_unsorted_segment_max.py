import tensorflow as tf

"""tf.unsorted_segment_max(data, segment_ids, num_segments, name=None)
功能：tensor进行拆分后求最大值。不同于sugementmax，segmentids不用按照顺序排列
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
    num_segments:分类总数，若多余ids匹配的数目，则置为numpy函数中可能的最小值。 """
""" If the maximum is empty for a given segment ID `i`, it outputs the smallest possible value for specific numeric type,
   `output[i] = numeric_limits<T>::min()`."""

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = tf.unsorted_segment_max(a, [0, 1, 0], 2)
z2 = tf.unsorted_segment_max(a, [0, 0, 0], 2)

sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
sess.close()
#
# [[7 8 9]
#  [4 5 6]]
# [[          7           8           9]
#  [-2147483648 -2147483648 -2147483648]]
