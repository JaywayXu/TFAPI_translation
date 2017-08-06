"""def split(value, num_or_size_splits, axis=0, num=None, name="split"):
  Splits a tensor into sub tensors.

  If `num_or_size_splits` is a scalar, `num_split`, then splits `value` along
  dimension `axis` into `num_split` smaller tensors.
  Requires that `num_split` evenly divides `value.shape[axis]`.

  If `num_or_size_splits` is a tensor, `size_splits`, then splits `value` into
  `len(size_splits)` pieces. The shape of the `i`-th piece has the same size as
  the `value` except along dimension `axis` where the size is `size_splits[i]`.
解释：这个函数的作用是，沿着split_dim维度将value切成num_split块。
For example:

  ```python
  # 'value' is a tensor with shape [5, 30]
  # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
  split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
  tf.shape(split0) ==> [5, 4]
  tf.shape(split1) ==> [5, 15]
  tf.shape(split2) ==> [5, 11]
  # Split 'value' into 3 tensors along dimension 1
  split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
  tf.shape(split0) ==> [5, 10]
   Args:
    value: The `Tensor` to split.
    num_or_size_splits: Either an integer indicating the number of splits along
      split_dim or a 1-D Tensor containing the sizes of each output tensor
      along split_dim. If an integer then it must evenly divide
      `value.shape[axis]`; otherwise the sum of sizes along the split
      dimension must match that of the `value`.
    axis: A 0-D `int32` `Tensor`. The dimension along which to split.
      Must be in the range `[0, rank(value))`. Defaults to 0.
    num: Optional, used to specify the number of outputs when it cannot be
      inferred from the shape of `size_splits`.
    name: A name for the operation (optional).
  """

import tensorflow as tf

sess = tf.Session()
input = tf.random_normal([5, 30])
print(sess.run(tf.shape(input))[0]/5)
# tf.shape(input)[0]=5
# tf.shape(input)[0]/5=1.0
split0, split1, split2, split3, split4 = tf.split(input, 5)
"""将input数据块从第0维开始分成5块"""
print(sess.run(tf.shape(split0)))
"""输入参数：
  ● split_dim: 一个0维的Tensor，数据类型是int32，
    该参数的作用是确定沿着哪个维度进行切割，参数范围 [0, rank(value))。
  ● num_split: 一个0维的Tensor，数据类型是int32，切割的块数量。
  ● value: 一个需要切割的Tensor。
  ● name:（可选）为这个操作取一个名字。
输出参数：
  ● 从value中切割的num_split个Tensor。"""
