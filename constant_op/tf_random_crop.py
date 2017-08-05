import tensorflow as tf

"""tf.random_crop(value, size, seed=None, name=None)
功能：将tensor按照指定大小进行随机裁剪。
输入：value：tensor;
    size：裁剪后的大小，size<=value.shape，如果不想改变大小，应配置为value的shape。"""

x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3, 3])
a = tf.random_crop(x, size=[2, 2], seed=11)

sess = tf.Session()
print(sess.run(a))
sess.close()
# a==>[[2 3]
#      [5 6]]
