"""tf.string_to_number(string_tensor, out_type = None, name = None)
解释：这个函数是将一个string的Tensor转换成一个数字类型的Tensor。
    但是要注意一点，如果你想转换的数字类型是tf.float32，那么这个string去掉引号之后，
    里面的值必须是一个合法的浮点数，否则不能转换。如果你想转换的数字类型是tf.int32，
    那么这个string去掉引号之后，里面的值必须是一个合法的浮点数或者整型，否则不能转换。"""

import tensorflow as tf

sess = tf.Session()
data = tf.constant("123")
print(sess.run(data))
d = tf.string_to_number(data)
print(sess.run(d))
"""输入参数：
  ● string_tensor: 一个string类型的Tensor。
  ● out_type: 一个可选的数据类型tf.DType，默认的是tf.float32，但我们也可以选择tf.int32或者tf.float32。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型是out_type，数据维度和string_tensor相同。"""