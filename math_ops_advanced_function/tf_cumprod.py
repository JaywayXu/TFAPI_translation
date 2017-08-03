import tensorflow as tf

"""tf.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)
功能：沿着维度axis进行累加。
输入：axis:默认为0
    reverse：默认为False，若为True，累加反向相反。"""

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = tf.cumprod(a)
z2 = tf.cumprod(a, axis=1)
z3 = tf.cumprod(a, reverse=True)

sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
sess.close()

# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]
# [[  1   2   6]
#  [  4  20 120]
#  [  7  56 504]]
# [[ 28  80 162]
#  [ 28  40  54]
#  [  7   8   9]]