"""class tf.TFRecordReader

A Reader that outputs the records from a TFRecords file.
从TFrecords文件中读取记录
See ReaderBase for supported methods.

tf.TFRecordReader.__init__(name=None)

Create a TFRecordReader.
创建一个TFRecordReader

Args:

name: A name for the operation (optional).
tf.TFRecordReader.num_records_produced(name=None)

Returns the number of records this reader has produced.

This is the same as the number of Read executions that have
succeeded.
返回这个阅读器生成的记录的数量。这与已成功执行读取操作的数量相同。

Args:

name: A name for the operation (optional).
Returns:

An int64 Tensor.
一个int64位张量.

tf.TFRecordReader.num_work_units_completed(name=None)

Returns the number of work units this reader has finished processing.
返回该阅读器完成处理的工作单元的数量。

Args:

name: A name for the operation (optional).
Returns:

An int64 Tensor.

tf.TFRecordReader.read(queue, name=None)

Returns the next record (key, value pair) produced by a reader.
返回一个阅读器生成的下一个记录(键值对)。
Will dequeue a work unit from queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has
finished with the previous file).
如果有必要，将从队列中对一个工作单元进行排序(例如，当读者需要从一个新文件开始阅读时，因为它已经完成了前面的文件)。

Args:

queue: A Queue or a mutable string Tensor representing a handle
to a Queue, with string work items.
文件名队列句柄
name: A name for the operation (optional).
Returns:

A tuple of Tensors (key, value).

key: A string scalar Tensor.
value: A string scalar Tensor.
返回键值对,其中值表示读取的文件
tf.TFRecordReader.reader_ref

Op that implements the reader.

tf.TFRecordReader.reset(name=None)

Restore a reader to its initial clean state.
恢复一个文件阅读器使其置空

Args:

name: A name for the operation (optional).
Returns:

The created Operation.

tf.TFRecordReader.restore_state(state, name=None)

Restore a reader to a previously saved state.
恢复阅读器至先前保存的状态.
Not all Readers support being restored, so this can produce an
Unimplemented error.
并不是所有的阅读器都可以实现恢复的操作,所以这有可能导致一个未实现的错误.

Args:

state: A string Tensor.
一个字符串张量
Result of a SerializeState of a Reader with matching type.
一个具有匹配类型的阅读器的串行化的结果。
name: A name for the operation (optional).
Returns:

The created Operation.

tf.TFRecordReader.serialize_state(name=None)

Produce a string tensor that encodes the state of a reader.
产生一个字符串张量，它可以对一个阅读器的状态进行编码。

Not all Readers support being serialized, so this can produce an
Unimplemented error.
不是所有的阅读器都支持编码,所以这会导致一个未实现的错误.

Args:

name: A name for the operation (optional).
Returns:

A string Tensor.

tf.TFRecordReader.supports_serialize

Whether the Reader implementation can serialize its state.
阅读器是否可以实现对当前状态进行编码.
"""