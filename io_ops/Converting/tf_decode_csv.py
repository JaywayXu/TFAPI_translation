"""tf.decode_csv(records, record_defaults, field_delim=None, name=None)

Convert CSV records to tensors. Each column maps to one tensor.
将CSV记录转换为张量。每一列都映射到一个张量。

Note that we allow leading and trailing spaces with int or float field.
请注意，我们允许带int或float字段的开头和尾附有空格。

Args:

records:
A Tensor of type string.
Each string is a record/row in the csv and all records should have
the same format.每个字符串都是csv中的记录/行，所有记录都应该具有相同的格式。(注意表示其中都是默认以字符串形式存储)


record_defaults:
A list of Tensor objects with types from: float32, int32, int64, string.
One tensor per column of the input record, with either a
scalar default value for that column or empty if the column is required.
输入记录的每一列的一个张量，用于表示空白列的默认标量值.当然这也指示了数据的默认类型!!

field_delim:
An optional string. Defaults to ",".
delimiter to separate fields in a record.分隔符来分隔记录中的字段。
name:
A name for the operation (optional).
Returns:

A list of Tensor objects. Has the same type as record_defaults.
Each tensor will have the same shape as records.
返回一个tensor对象的列表,并且每个和"record_defaults"含有相同的类型,作为结果的每个张量会有相同的shape"""
