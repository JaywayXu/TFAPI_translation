"""保存一个模型"""
# import tensorflow as tf
#
# v = tf.Variable(0, dtype=tf.float32, name='v')
# v1 = tf.Variable(0, dtype=tf.float32, name='v1')
#
# result = tf.add(v, v1)
#
# x = tf.placeholder(tf.float32, shape=[1], name='x')
#
# test = tf.add(result, x)
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     saver.save(sess, "./Files/model_2.ckpt")
"""使用tf.train.NewCheckpointReader导出所有变量"""
import tensorflow as tf

reader = tf.train.NewCheckpointReader("./Files/model_2.ckpt")

variables = reader.get_variable_to_shape_map()  #此处将variable转换为string类型

for ele in variables:
    print(ele)