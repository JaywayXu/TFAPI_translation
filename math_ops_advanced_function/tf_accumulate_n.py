import tensorflow as tf

"""tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)
功能：对应位置元素相加。如果输入是训练变量，不要使用，应使用tf.add_n。
输入：shape，tensor_dtype:类型检查"""
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
z = tf.accumulate_n([a, b])

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[6 8]
#      [10 12]]
