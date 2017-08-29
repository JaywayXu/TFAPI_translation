"""tf.pack(values, axis=0, name=”pack”)
Packs a list of rank-R tensors into one rank-(R+1) tensor
将一个R维张量列表沿着axis轴组合成一个R+1维的张量。
  Given a list of length `N` of tensors of shape `(A, B, C)`;
  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.
特别注意pack函数已经更名为stack函数."""
import tensorflow as tf

x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
with tf.Session() as sess:
    print(sess.run(tf.stack([x, y, z])))
    print(sess.run(tf.stack([x, y, z], axis=1)))
# [[1 4]
#  [2 5]
#  [3 6]]
# [[1 2 3]
#  [4 5 6]]