"""从CSV文件中读取数据,需要使用TextLineReader和decode_csv 操作， 如下面的例子所示：
每次read的执行都会从文件中读取一行内容， decode_csv 操作会解析这一行内容并将其转为张量列表。
如果输入的参数有缺失，record_default参数可以根据张量的类型来设置默认值。"""
# -*- coding: utf-8 -*-
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# 设置列的默认值
# decoded result.是一个极其关键的值
record_defaults = [["string"], [""], [""], [""], [""]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
data_1 = [col1, col2, col3, col4]
with tf.Session() as sess:
    print(sess.run(tf.shape(data_1)))  # [4]
    print(sess.run(tf.shape(col1)))  # []

features = [col1, col2, col3, col4, col5]
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()  # 创建线程管理器
    threads = tf.train.start_queue_runners(coord=coord)
    # 在调用run或者eval去执行read之前,必须先调用tf.train.start_queue_runners来讲文件名补充到队列中,
    # 否则read操作会被最到文件名队列中直到有值为止

    for i in range(2):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print(example)
        print(label)

    coord.request_stop()  # 所有的线程都可以调用coord.request_stop()函数来是所有的线程中止
    coord.join(threads)
    # [b'\xe6\xad\xbb\xe7\xa5\x9e', b'\xe7\x81\xab\xe5\xbd\xb1', b'\xe6\xb5\xb7\xe8\xb4\xbc', b'EVA',
    #  b'\xe5\x9c\xa3\xe6\x96\x97\xe5\xa3\xab']
    # b'\xe5\x9c\xa3\xe6\x96\x97\xe5\xa3\xab'

    # [b'\xe6\x96\xa9\xe6\x9c\x88', b'\xe9\x9c\xb2\xe7\x90\xaa\xe4\xba\x9a', b'\xe9\x98\xbf\xe6\x95\xa3\xe4\xba\x95',
    # b'\xe7\x99\xbd\xe5\x93\x89', b'\xe5\xb0\x8f\xe7\x99\xbd']
    # b'\xe5\xb0\x8f\xe7\x99\xbd'
