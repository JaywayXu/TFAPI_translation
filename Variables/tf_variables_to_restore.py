"""
variables_to_restore是为了在保持模型的时候方便使用滑动平均的参数，如果不使用这个保存，那模型就会保存所以参数，
除非你提前设定，就是在保存的时候指定保存变量也是可以的，比如saver = tf.train.Saver([v])这样就可以指定保存变量v，
在模型导入的时候只有这个变量会被导入。
"""
import tensorflow as tf

# v = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='v')
# ema = tf.train.ExponentialMovingAverage(0.99)  # 设定滑动平均函数
# maintain_average_op = ema.apply(tf.global_variables())  # 将所有变量值加入滑动平均函数中
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     sess.run(tf.assign(v, 10.0))
#     sess.run(maintain_average_op)
#     saver.save(sess, './Files/models_01.ckpt')
# 以上是变量的保存阶段,这时候变量被保存到ckpt文件中,我们主要做的是ckpt的restore操作

"""模型导入_1"""

# v = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='v')
#
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_average_op = ema.apply(tf.global_variables())
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, './Files/models_01.ckpt')
#     print(sess.run(ema.average(v)))
#     sess.run(ema.average(v))
#     print(sess.run(v))
# 0.0999999
# 10.0
"""这样不是很方便，因为我再次导入模型，变量v的值我不用，并且想要用计算后的值替代v，
而此处还是v原先的值,我们需要输出函数经过计算后的值"""

"""模型导入_2"""

"""导入模型的时候tf.train.Saver函数要变化一下，变为tf.train.Saver(ema.variables_to_restore()"""
v = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='v')

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())

saver = tf.train.Saver(ema.variables_to_restore())  # 这是导入计算后的值的关键步骤
with tf.Session() as sess:
    saver.restore(sess, './Files/models_01.ckpt')
    print(sess.run(v))
    # 0.0999999
