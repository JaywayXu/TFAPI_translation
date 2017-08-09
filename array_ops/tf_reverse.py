"""tf.reverse(tensor, dims, name = None)
解释：将指定维度中的数据进行翻转。
给定一个tensor和一个bool类型的dims，dims中的值为False或者True。
如果dims[i] == True，那么就将tensor中这一维的数据进行翻转。
tensor最多只能有8个维度，并且tensor的秩必须和dims的长度相同，
即rank(tensor) == size(dims)。"""

import tensorflow as tf

sess = tf.Session()
input_data = tf.constant([[
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ],
    [
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]
    ]
]])
print('input_data shape : ', sess.run(tf.shape(input_data)))
# [1, 2, 3, 4]
dims = tf.constant([3])
print(sess.run(tf.reverse(input_data, dims)))
# [[[[ 3  2  1  0]
#    [ 7  6  5  4]
#    [11 10  9  8]]
#
#   [[15 14 13 12]
#    [19 18 17 16]
#    [23 22 21 20]]]] 将第三维也就是最后一维的数据进行了翻转操作
print("==========================")
dims = tf.constant([1])
print(sess.run(tf.reverse(input_data, dims)))
# [[[[12 13 14 15]
#    [16 17 18 19]
#    [20 21 22 23]]
#
#   [[ 0  1  2  3]
#    [ 4  5  6  7]
#    [ 8  9 10 11]]]]  将第一维也就是上下两个部分的数据进行了调换
print("==========================")
dims = tf.constant([2])
print(sess.run(tf.reverse(input_data, dims)))
# [[[[ 8  9 10 11]
#    [ 4  5  6  7]
#    [ 0  1  2  3]]
#
#   [[20 21 22 23]
#    [16 17 18 19]
#    [12 13 14 15]]]] 将两个部分中的三行都进行了调换
sess.close()
"""输入参数：
  ● tensor: 一个Tensor，数据类型必须是以下之一：uint8，int8，int32，bool，float32或者float64，数据维度不超过8维。
  ● dims: 一个Tensor，数据类型是bool。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和tensor相同，数据维度和tensor相同。"""
