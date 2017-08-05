import tensorflow as tf

"""tf.multinomial(logits, num_samples, seed=None, name=None)
功能：绘制多项式分布。
输入：logits：shape为[batch_size,num_classes]的2维tensor，每行[i，：]代表每类出现的概率。
     形状是[批数量,类别数目]
     num_samples：独立采样数目。
Returns:
    The drawn samples of shape `[batch_size, num_samples]`."""
"""设定有四批数据，每批数据都有三种类型"""
x = tf.constant([[1, 2, 100], [1, 5, 100], [1, 2, 100], [1, 3, 100]], dtype=tf.float32)
a = tf.multinomial(x, 5)
sess = tf.Session()
print(sess.run(a))
# [[2 2 2 2 2]
#  [2 2 2 2 2]
#  [2 2 2 2 2]
#  [2 2 2 2 2]]
"""上面的例子相当于我设置了3个类别，并且在每批数据中都设置第三类（2，因为类别数组从0开始计算的缘故）的出现的概率最大
（有100，相对于第1,2类的数据的出现概率大了几乎；两个数量级）"""
y = tf.constant([[100, 1, 1], [1, 100, 1], [1, 1, 100], [1, 100, 1]], dtype=tf.float32)
b = tf.multinomial(y, 9)
print(sess.run(b))
sess.close()

# [[0 0 0 0 0 0 0 0 0]
#  [1 1 1 1 1 1 1 1 1]
#  [2 2 2 2 2 2 2 2 2]
#  [1 1 1 1 1 1 1 1 1]]


