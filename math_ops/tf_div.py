"""tf.div(x, y, name = None)
解释：这个函数返回x与y逐元素相除的结果。
使用例子："""


import tensorflow as tf

a = tf.constant([1, 2])
b = tf.constant(2)
c = tf.div(a, b)
sess = tf.Session()
print(sess.run(c))  # [0 1]
sess.close()

a0 = tf.constant([2, 4])
b0 = tf.constant([1, 2])
c0 = tf.div(a0, b0)
sess = tf.Session()
print(sess.run(c0))  # [2 2]
sess.close()

a1 = tf.constant([[2, 4], [3, 6]])
b1 = tf.constant([1, 2])
c1 = tf.div(a1, b1)
sess = tf.Session()
print(sess.run(c1))
# [[2 2]
#  [3 3]]
sess.close()
"""输入参数：
  ● x: 一个Tensor，数据类型是必须是以下之一：float32，float64，int8，int16，int32，complex64，int64。
  ● y: 一个Tensor，数据类型必须和x相同。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 一个Tensor，数据类型和x相同。"""
"""
PS:
declaration->div
    def div(x, y, name=None):
      Divides x / y elementwise (using Python 2 division operator semantics).
      使用Python 2除法运算符语义
    
      NOTE: Prefer using the Tensor division operator or tf.divide which obey Python
      division operator semantics.
      推荐使用Tensor分割运算符或tf.divide，它遵守Python分割运算符语义。
      This function divides `x` and `y`, forcing Python 2.7 semantics. That is,
      if one of `x` or `y` is a float, then the result will be a float.
      Otherwise, the output will be an integer type. Flooring semantics are used
      for integer division.
    
      Args:
        x: `Tensor` numerator of real numeric type.
        y: `Tensor` denominator of real numeric type.
        name: A name for the operation (optional).
      Returns:
        `x / y` returns the quotient of x and y.
      
      return _div_python2(x, y, name)


    declaration->gen_math_ops._sub
        def _sub(x, y, name=None):
            result = _op_def_lib.apply_op("Sub", x=x, y=y, name=name)
            return result
"""