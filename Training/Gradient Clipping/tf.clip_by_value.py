import tensorflow as tf
import numpy as np

# tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
# 小于min的让它等于min，大于max的元素的值等于max。
A = np.array([[1, 1, 2, 4], [3, 4, 8, 5]])
with tf.Session()as sess:
    print(sess.run(tf.clip_by_value(A, 2, 5)))
#
# [[2 2 2 4]
#  [3 4 5 5]]