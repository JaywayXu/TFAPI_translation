import tensorflow as tf

"""tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
功能：沿着维度axis进行累加。
输入：axis:默认为0
    reverse：默认为False，若为True，累加反向相反。"""

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = tf.cumsum(a)
z2 = tf.cumsum(a, axis=1)
z3 = tf.cumsum(a, reverse=True)

sess = tf.Session()

print(sess.run(z))
print(sess.run(z2))
print(sess.run(z3))
sess.close()

# z==>[[1 2 3]
#      [5 7 9]
#      [12 15 18]]
# z2==>[[1 3 6]
#       [4 9 15]
#       [7 15 24]]
# z3==>[[12 15 18]
#       [11 13 15]
#       [7 8 9]]
