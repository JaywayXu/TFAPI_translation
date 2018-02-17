""" 多项式学习率衰减
特点是确定结束的学习率。
polynomial_decay(learning_rate, global_step, decay_steps,
                     end_learning_rate=0.0001, power=1.0,
                     cycle=False, name=None):
通常观察到，通过仔细选择的变化程度的单调递减的学习率会产生更好的表现模型。此函数将多项式衰减应用于学习率的初始值。
使学习率`learning_rate`在给定的`decay_steps`中达到`end_learning_rate`。
它需要一个`global_step`值来计算衰减的学习速率。你可以传递一个TensorFlow变量，在每个训练步骤中增加
  global_step = min(global_step, decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

如果`cycle`为True，则使用`decay_steps`的倍数，第一个大于'global_steps`.ceil表示向上取整。
  decay_steps = decay_steps * ceil(global_step / decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

"""

''' Example: decay from 0.1 to 0.01 in 10000 steps using sqrt (i.e. power=0.5):'''
import tensorflow as tf
import matplotlib.pyplot as plt

global_ = tf.Variable(tf.constant(0), trainable=False)
starter_learning_rate = 0.1  # 初始学习率
end_learning_rate = 0.01  # 结束学习率
decay_steps = 1000
globalstep = 10000
f = tf.train.polynomial_decay(starter_learning_rate, global_, decay_steps, end_learning_rate, power=0.5, cycle=False)
t = tf.train.polynomial_decay(starter_learning_rate, global_, decay_steps, end_learning_rate, power=0.5, cycle=True)
F = []
T = []
with tf.Session() as sess:
    for i in range(globalstep):
        f_ = sess.run(f, feed_dict={global_: i})
        F.append(f_)
        t_ = sess.run(t, feed_dict={global_: i})
        T.append(t_)

plt.figure(1)
plt.plot(range(globalstep), F, 'r-')
plt.plot(range(globalstep), T, 'b-')
plt.show()
