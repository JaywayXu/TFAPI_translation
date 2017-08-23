"""Session.run方法接受一个参数fetches,以及其它三个可选参数feed_dict/option/run_metadata"""

"""
fetches
fetches参数接受任意的数据流图元素(OP或Tensor),或者指定了用户希望执行的对象,
如果请求对象是Tensor对象,则run()的输出将会是Numpy数组,如果请求对象是一个OP,则输出结果将会是None
当fetches为一个列表式,run()的输出将会是一个与所请求的元素对应的值的列表"""

"""
feed_dict
参数feed_dict用于覆盖数据刘涂红的Tensor对象值,它需要Python字典对象作为输入.
字典中的"键"作为指向相应被覆盖的tensor对象句柄,儿子店中的"值"可以是数字,字符串,列表或者Numpy数组.
"""
# 下面将展式如何利用feed_dict重写之前的数据
import tensorflow as tf

# 创建OP/tensor对象等(使用默认的数据流图)
a = tf.add(2, 5)
b = tf.multiply(2, 3)
# 利用默认的数据流图启动一个Sessio对象
sess = tf.Session()
# 定义一个字典,比如将a的值替换为15
replace_dict = {a: 15}
# 运行Session独享,将replace_dict赋值给feed_dict
sess.run(b, feed_dict=replace_dict)
"""
由于张量的值是预先提供的,数据流图不再需要对该张量的任何普通节点进行计算,这意味着如果有一个规模较大的数据流图,
并且希望用一些虚构的值对某些部分进行测试,tensorflow将不会在不必要的计算上浪费时间.

"""
"""
在Session对象使用完毕后,需要调用close方法,将不再需要的资源释放.sess.close()
"""
"""
也可以将Session类作为上下文管理器加以使用,这样代码离开作用域后,该Session对象将自动关闭
"""
"""
with tf.Session() as sess:
      # 运行数据流图,写入概括计量
      ...
# Session 对象自动关闭
"""
"""
也可利用Session类的as_default()方法将Session对象作为上下文管理器加以使用.
可将一个Session对象设置为可悲某些函数自动调用.这些函数中常见的有Operation.run()和Tensor.eval().
调用这些函数可以将他们直接传入Session.run函数.
"""
# 定义简单的常量
a = tf.constant(5)
# 创建一个Session对象
sess = tf.Session()
# 在with语句块中将该Session对象作为默认的Session对象
with sess.as_default():
    a.eval()
# 必须手动关闭Session对象
sess.close()


