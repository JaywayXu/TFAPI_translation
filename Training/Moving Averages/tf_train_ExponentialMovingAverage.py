import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)  # 定义一个变量，初始值为0
step = tf.Variable(0, trainable=False)  # step为迭代轮数变量，控制衰减率
ema = tf.train.ExponentialMovingAverage(0.99, step)  # 初始设定衰减率为0.99
maintain_averages_op = ema.apply([v1])  # 更新列表中的变量
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化所有变量
sess.run(init_op)
print(sess.run([v1, ema.average(v1)]))  # 输出初始化后变量v1的值和v1的滑动平均值
sess.run(tf.assign(v1, 5))  # 更新v1的值
sess.run(maintain_averages_op)  # 更新v1的滑动平均值
print(sess.run([v1, ema.average(v1)]))
sess.run(tf.assign(step, 10000))  # 更新迭代轮转数step
sess.run(tf.assign(v1, 10))
sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
# 再次更新滑动平均值，
sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
# 更新v1的值为15
sess.run(tf.assign(v1, 15))

sess.run(maintain_averages_op)
print(sess.run([v1, ema.average(v1)]))
#
# [0.0, 0.0]
# [5.0, 4.5]
# [10.0, 4.5549998]
# [10.0, 4.6094499]
# [15.0, 4.7133551]