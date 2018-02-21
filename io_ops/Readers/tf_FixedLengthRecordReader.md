## class tf.FixedLengthRecordReader
A Reader that outputs fixed-length records from a file.
从文件输出固定长度记录的阅读器。
See ReaderBase for supported methods.
请参阅ReaderBase支持的方法。

## tf.FixedLengthRecordReader.__init__(record_bytes, header_bytes=None, footer_bytes=None, name=None)
Create a FixedLengthRecordReader.
创建一个匹配长度的记录文件阅读器。

**Args:**
record_bytes: An int.
header_bytes: An optional int. Defaults to 0.
footer_bytes: An optional int. Defaults to 0.
name: A name for the operation (optional).

## tf.FixedLengthRecordReader.num_records_produced(name=None)

Returns the number of records this reader has produced.
返回该读取器生成的记录数。
This is the same as the number of Read executions that have
succeeded.
这与成功执行的读执行次数相同。
**Args:**
name: A name for the operation (optional).
Returns:
An int64 Tensor.

## tf.FixedLengthRecordReader.num_work_units_completed(name=None)
Returns the number of work units this reader has finished processing.
返回该阅读器已完成处理的工作单元数。

**Args:**
name: A name for the operation (optional).
Returns:
An int64 Tensor.

## tf.FixedLengthRecordReader.read(queue, name=None)
Returns the next record (key, value pair) produced by a reader.
返回阅读器读取的下一个键值对记录文件
Will dequeue a work unit from queue if necessary (e.g. when the Reader needs to start reading from a new file since it has finished with the previous file).
如果需要，将从队列中删除一个工作单元(例如，当读者需要从一个新文件开始阅读时，因为它已经完成了前面的文件)。

**Args:**
queue: A Queue or a mutable string Tensor representing a handle to a Queue, with string work items.
队列或可变字符串张量，表示队列的句柄，带有字符串工作项。
name: A name for the operation (optional).
Returns:
A tuple of Tensors (key, value).
一组键值对张量

key: A string scalar Tensor.
value: A string scalar Tensor.

## tf.FixedLengthRecordReader.reader_ref
Op that implements the reader.
实现阅读器的函数。

## tf.FixedLengthRecordReader.reset(name=None)
Restore a reader to its initial clean state.
使阅读器恢复到初始化状态。

**Args:**
name: A name for the operation (optional).
Returns:
The created Operation.

## tf.FixedLengthRecordReader.restore_state(state, name=None)
Restore a reader to a previously saved state.
将阅读区恢复到先前保存的状态。
Not all Readers support being restored, so this can produce an Unimplemented error.
并不是所有的阅读器都支持恢复，所以这可能产生一个未实现的错误。

**Args:**
state: A string Tensor.
Result of a SerializeState of a Reader with matching type.
具有匹配类型的阅读器的序列化状态。
name: A name for the operation (optional).
Returns:
The created Operation.

## tf.FixedLengthRecordReader.serialize_state(name=None)
Produce a string tensor that encodes the state of a reader.
生成一个字符串张量，用于编码阅读器的状态。
Not all Readers support being serialized, so this can produce an Unimplemented error.
并不是所有的阅读器支持被序列化，因此这可能产生一个未实现的错误。

**Args:**
name: A name for the operation (optional).
Returns:
A string Tensor.
字符串类型张量

## tf.FixedLengthRecordReader.supports_serialize
Whether the Reader implementation can serialize its state.
