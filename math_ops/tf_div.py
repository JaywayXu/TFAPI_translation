"""tf.div(x, y, name = None)
解释：这个函数返回x与y逐元素相除的结果。现在已经不推荐该使用方法，推荐你使用tf.divide()
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
"""
tf.div(x,y,name=None)[推荐使用tf.divide(x,y)]
功能：对应位置元素的除法运算（使用python2.7除法算法，如果x,y有一个为浮点数，结果为浮点数;否则为整数，但使用该函数会报错）。
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。
例：
x=tf.constant([[1,4,8]],tf.int32)
y=tf.constant([[2,3,3]],tf.int32)
z=tf.div(x,y)

z==>[[0,1,2]]

x=tf.constant([[1,4,8]],tf.int64)
y=tf.constant([[2,3,3]],tf.int64)
z=tf.divide(x,y)

z==>[[0.5,1.33333333,2.66666667]]

x=tf.constant([[1,4,8]],tf.float64)
y=tf.constant([[2,3,3]],tf.float64)
z=tf.div(x,y)

z==>[[0.5,1.33333333,2.66666667]]
"""
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
      这个函数根据python2.7语义，也就是如果x或者y是一个float类型的，结果也会是一个float类型的，否则结果会是一个int类型的数
    
      Args:
        x: `Tensor` numerator of real numeric type.
        y: `Tensor` denominator of real numeric type.
        name: A name for the operation (optional).
      Returns:
        `x / y` returns the quotient of x and y.
      
      return _div_python2(x, y, name)

    declaration->_div_python2
        def _div_python2(x, y, name=None):
          Divide two values using Python 2 semantics. Used for Tensor.__div__.
        
          Args:
            x: `Tensor` numerator of real numeric type.(分子)
            y: `Tensor` denominator of real numeric type.（分母）
            name: A name for the operation (optional).
          Returns:
            `x / y` returns the quotient of x and y.
  

        with ops.name_scope(name, "div", [x, y]) as name:
            x = ops.convert_to_tensor(x, name="x")
            y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
            x_dtype = x.dtype.base_dtype
            y_dtype = y.dtype.base_dtype
            if x_dtype != y_dtype:
              raise TypeError("x and y must have the same dtype, got %r != %r" %
                              (x_dtype, y_dtype))
            if x_dtype.is_floating or x_dtype.is_complex:
              return gen_math_ops._real_div(x, y, name=name)  # real_div
            else:
              return gen_math_ops._floor_div(x, y, name=name)  # floor_div
"""