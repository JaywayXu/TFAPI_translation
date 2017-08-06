"""tf.slice(input_, begin, size, name = None)
解释：这个函数的作用是从输入数据input中提取出一块切片，切片的尺寸是size，
切片的开始位置是begin。切片的尺寸size表示输出tensor的数据维度，
其中size[i]表示在第i维度上面的元素个数。开始位置begin表示切片相对于输入数据input_的每一个偏移量，比如数据input_是
[[[1, 1, 1], [2, 2, 2]],
[[33, 3, 3], [4, 4, 4]],
[[5, 5, 5], [6, 6, 6]]]，
begin为[1, 0, 0]，那么数据的开始位置是33。因为，第一维偏移了1，其余几位都没有偏移，所以开始位置是33。
操作满足：
size[i] = input.dim_size(i) - begin[i]
0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n]"""

import tensorflow as tf

sess = tf.Session()
input = tf.constant([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]])
data = tf.slice(input, [1, 0, 0], [1, 1, 3])
print(sess.run(data))
"""[1,0,0]表示第一维偏移了1
则是从[[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]中选取数据
然后选取第一维的第一个，第二维的第一个数据，第三维的三个数据"""
# [[[3 3 3]]]
data = tf.slice(input, [1, 0, 0], [1, 2, 3])
print(sess.run(data))
# [[[3 3 3]
#   [4 4 4]]]
data = tf.slice(input, [1, 0, 0], [2, 1, 3])
print(sess.run(data))
# [[[3 3 3]]
#
#  [[5 5 5]]]
data = tf.slice(input, [1, 0, 0], [2, 2, 2])
print(sess.run(data))
# [[[3 3]
#   [4 4]]
#
#  [[5 5]
#   [6 6]]]
"""输入参数：
  ● input_: 一个Tensor。
  ● begin: 一个Tensor，数据类型是int32或者int64。
  ● size: 一个Tensor，数据类型是int32或者int64。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和input_相同。"""
