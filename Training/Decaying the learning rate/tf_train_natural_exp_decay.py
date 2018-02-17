"""学习率自然指数衰减
def natural_exp_decay(learning_rate, global_step, decay_steps, decay_rate,
                      staircase=False, name=None)
将自然指数衰减应用于初始学习速率。
在训练模型时，经常建议在训练过程中降低学习速度。该函数将指数衰减函数应用于提供的初始学习速率。
它需要一个`global_step`值来计算衰减的学习速率。你可以传递一个TensorFlow变量，在每个训练步骤中增加
decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
"""
import tensorflow as tf
import matplotlib.pyplot as plt

global_ = tf.Variable(tf.constant(0), trainable=False)
globalstep = 10000  # 全局下降步数
learning_rate = 0.1  # 初始学习率
decaystep = 1000  # 实现衰减的频率
decay_rate = 0.5  # 衰减率

t = tf.train.natural_exp_decay(learning_rate, global_, decaystep, decay_rate, staircase=True)
f = tf.train.natural_exp_decay(learning_rate, global_, decaystep, decay_rate, staircase=False)

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
