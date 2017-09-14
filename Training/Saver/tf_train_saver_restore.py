import tensorflow as tf

v1 = tf.Variable(tf.constant(0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(0, shape=[1]), name='v2')
v = [v1, v2]
result = tf.add_n(v)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./Files/model_01.ckpt")
    print(sess.run(result))
    # [3]

    # 注意这里只是重新定义了图的结构,但是没有使用函数
    # init = tf.initialize_all_variables()
    # sess.run(init)
    # 对变量进行初始化操作,也就是说这里并没有使用此处定义的v1和v2的值
