"""TFRecords文件表示一个(二进制)字符串序列。格式不支持随机访问，因此它适合于大量的数据流，但不
适用于快速分片或其他非连续存取。"""
"""tf.python_io.TFRecordWriter.write(record)

Write a string record to the file.
将字符记录写到文件中,注意传入的参数是string类型的字符串.

Args:

record: str"""