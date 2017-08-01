"""tf.subtract(x, y, name = None)
解释：这个函数返回x与y逐元素相减的结果。
使用例子："""
import tensorflow as tf
# 减数是一个数
a = tf.constant([1, 2, 3])
b = tf.constant(2)
c = tf.subtract(a, b)
sess = tf.Session()
print(sess.run(c))  # [-1 0 1]
sess.close()

# 不同维度的数组相减
# a1 = tf.constant([1, 2, 2, 2])
# b1 = tf.constant([2, 2])
# c1 = tf.subtract(a1, b1)
# # ValueError: Dimensions must be equal, but are 4 and 2 for 'Sub_1' (op: 'Sub') with input shapes: [4], [2].
# sess = tf.Session()
# print(sess.run(c1))
# sess.close()

# 高维数组减去常数
# a2 = tf.constant([[1, 2, 3], [2, 2, 2]])
# b2 = tf.constant(2)
# c2 = tf.subtract(a2, b2)
# sess = tf.Session()
# print(sess.run(c2))
# # [[-1  0  1]
# # [ 0  0  0]]
# sess.close()

#高维数组相减的条件
# a3 = tf.constant([[1, 2, 3], [2, 2, 2]])
# b3 = tf.constant([3, 3, 3])
# c3 = tf.subtract(a3, b3)
# sess = tf.Session()
# print(sess.run(c3))
# # [[-2 -1  0]
# #  [-1 -1 -1]]
# sess.close()

"""输入参数：
  ● x: 一个Tensor，数据类型是必须是以下之一：float32，float64，int8，int16，int32，complex64，int64。
  ● y: 一个Tensor，数据类型必须和x相同。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和x相同。"""
"""
PS:
declaration->subtract
    return gen_math_ops._sub(x, y, name)
    "2016-12-30",
    "`tf.sub(x, y)` is deprecated, please use `tf.subtract(x, y)` or `x - y`")

    declaration->gen_math_ops._sub
        def _sub(x, y, name=None):
            result = _op_def_lib.apply_op("Sub", x=x, y=y, name=name)
            return result
"""