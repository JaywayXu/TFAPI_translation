"""tf.train.match_filenames_once(pattern, name=None)

Save the list of files matching pattern, so it is only computed once.
保存文件匹配模式的列表，因此只计算一次。 用于产生文件名列表.

Args:

pattern: A file pattern (glob).
一个文件模式
name: A name for the operations (optional) or 1D tensor of file patterns.
Returns:

A variable that is initialized to the list of files matching pattern.
一个被初始化到文件匹配模式列表的变量。显然此函数调用的是name_scope函数,所以我们现在需要使用name_scope函数
函数定义:
def match_filenames_once(pattern, name=None):
  with ops.name_scope(name, "matching_filenames", [pattern]) as name:
    return vs.variable(
        name=name, initial_value=io_ops.matching_files(pattern),
        trainable=False, validate_shape=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])
对于io_ops.matching_files函数  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.
  注意，这个例程只支持模式中basename部分中的通配符字符，而不是在目录部分中

"""


import tensorflow as tf

directory = "cats_vs_dogs.tfrecords"
file_names = tf.train.match_filenames_once(directory)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())
# 初始化变量

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(file_names))
