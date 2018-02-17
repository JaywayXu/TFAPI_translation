"""常数分片学习率衰减

piecewise_constant(x, boundaries, values, name=None)
Example: use a learning rate that's 1.0 for the first 10000 steps, 0.5
    for steps 10001 to 12000, and 0.1 for any additional steps.
    例如前1W轮迭代使用1.0作为学习率，1W轮到1.1W轮使用0.5作为学习率，以后使用0.1作为学习率。

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 当global_取不同的值时learning_rate的变化，所以我们把global_
global_ = tf.Variable(tf.constant(0), trainable=False)
boundaries = [10000, 12000]
values = [1.0, 0.5, 0.1]
learning_rate = tf.train.piecewise_constant(global_, boundaries, values)
global_steps = 20000

T_L = []
with tf.Session() as sess:
    for i in range(global_steps):
        T_l = sess.run(learning_rate, feed_dict={global_: i})
        T_L.append(T_l)

plt.figure(1)
plt.plot(range(global_steps), T_L, 'r-')
plt.show()
