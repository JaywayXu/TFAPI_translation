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
