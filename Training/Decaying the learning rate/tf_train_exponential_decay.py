"""自适应学习率衰减

tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
退化学习率,衰减学习率,将指数衰减应用于学习速率。
计算公式:decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)

"""
# 初始的学习速率是0.1，总的迭代次数是1000次，如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率，
# 如果是False，那就是每一步都更新学习速率。红色表示False，蓝色表示True。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1  # 初始学习速率时0.1
decay_rate = 0.96  # 衰减率
global_steps = 1000  # 总的迭代次数
decay_steps = 100  # 衰减次数

global_ = tf.Variable(tf.constant(0))
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)

plt.figure(1)
plt.plot(range(global_steps), F_D, 'r-')# "-"表示折线图,r表示红色,b表示蓝色
plt.plot(range(global_steps), T_C, 'b-')
# 关于函数的值的计算0.96^(3/1000)=0.998
plt.show()
