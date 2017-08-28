"""tf.TextLineReader.read(queue, name=None)

Returns the next record (key, value pair) produced by a reader.
返回阅读器读取的下一条记录(键值对形式)

Will dequeue a work unit from queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has
finished with the previous file).
如果有必要，将对队列中的工作单元进行排序(例如,阅读器已经完成了一部分内容,现在需要从下一条进行阅读时)。

Args:

queue: A Queue or a mutable string Tensor representing a handle
to a Queue, with string work items.
代表文件名队列的句柄
name: A name for the operation (optional).
Returns:

A tuple of Tensors (key, value).
返回键值对

key: A string scalar Tensor.
value: A string scalar Tensor."""