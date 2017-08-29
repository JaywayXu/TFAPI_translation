import tensorflow as tf
"""tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, name=None)

Output strings (e.g. filenames) to a queue for an input pipeline.
对于输入的文件名列表,此函数会生成一个先入先出的队列,文件阅读器利用其来读取数据.
可以通过提供的可配置参数来设置文件名乱序和最大的训练迭代数

Args:

string_tensor: A 1-D string tensor with the strings to produce.
文件名序列
num_epochs: An integer (optional). If specified, string_input_producer
produces each string from string_tensor num_epochs times before
generating an OutOfRange error. If not specified, string_input_producer
can cycle through the strings in string_tensor an unlimited number of
times.
int类型数据,如果指定在num_epochs指定次数内读取文件名列表的文件名,如果不指定则会无限次循环读取文件名
shuffle: Boolean. If true, the strings are randomly shuffled within each
epoch.
布尔类型数据,如果设置为真,会对每一epoch中得数据进行乱序处理
seed: An integer (optional). Seed used if shuffle == True.
用于乱序处理的seed
capacity: An integer. Sets the queue capacity.
int类型,设置队列的容量
name: A name for the operations (optional)."""
