import tensorflow as tf

"""tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)
功能：tensor进行拆分后求和。不同于sugementsum，segmentids不用按照顺序排列
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
    num_segments:分类总数，若多余ids匹配的数目，则置0。"""

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = tf.unsorted_segment_sum(a, [0, 1, 0], 2)
z2 = tf.unsorted_segment_sum(a, [0, 0, 0], 2)

sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
sess.close()

# z==>[[8 10 12]
#      [4  5  6]]
# z2==>[[12 15 18]
#       [0  0  0]]
