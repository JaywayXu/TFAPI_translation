"""反时限学习率衰减
inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
将反时限衰减应用到初始学习率。
decayed_learning_rate = learning_rate / (1 + decay_rate * t)
"""

import tensorflow as tf
import matplotlib.pyplot as plt

global_ = tf.Variable(tf.constant(0), trainable=False)
globalstep = 10000  # 全局下降步数
learning_rate = 0.1  # 初始学习率
decaystep = 1000  # 实现衰减的频率
decay_rate = 0.5  # 衰减率

t = tf.train.inverse_time_decay(learning_rate, global_, decaystep, decay_rate, staircase=True)
f = tf.train.inverse_time_decay(learning_rate, global_, decaystep, decay_rate, staircase=False)

T = []
F = []

with tf.Session() as sess:
    for i in range(globalstep):
        t_ = sess.run(t, feed_dict={global_: i})
        T.append(t_)
        f_ = sess.run(f, feed_dict={global_: i})
        F.append(f_)

plt.figure(1)
plt.plot(range(globalstep), T, 'r-')
plt.plot(range(globalstep), F, 'b-')
plt.show()