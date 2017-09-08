"""tf.decode_raw(bytes, out_type, little_endian=None, name=None)

Reinterpret the bytes of a string as a vector of numbers.
将字符串的字节重新解释为一个数字的向量。

Args:

bytes: A Tensor of type string.
All the elements must have the same length.
string类型的张量所有的元素必须具有相同的长度
out_type: A tf.DType from: tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64.
little_endian: An optional bool. Defaults to True.
Whether the input bytes are in little-endian order.
Ignored for out_types that are stored in a single byte like uint8.
是否是小尾存储,小尾存储和大尾存储即’little_endian’&'big_endian’是数据在计算机内存中存储的不同的方式.现有的大部分计算机CPU的架构都是X86形式的,这样都是符合小尾存储的表示形式的
name: A name for the operation (optional).
Returns:

A Tensor of type out_type.
A Tensor with one more dimension than the input bytes. The
added dimension will have size equal to the length of the elements
of bytes divided by the number of bytes to represent out_type.
一个比输入字节多一个维度的张量。添加的维数将等于字节元素的长度除以字节数，以表示out_type
"""