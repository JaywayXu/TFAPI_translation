import tensorflow as tf

"""tf.where(condition, x=None, y=None, name=None)
功能：若x,y都为None，返回condition值为True的坐标;
    若x,y都不为None，返回condition值为True的坐标在x内的值，condition值为False的坐标在y内的值
输入：condition:bool类型的tensor;"""

a = tf.constant([True, False, False, True])
x = tf.constant([1, 2, 3, 4])
y = tf.constant([5, 6, 7, 8])
z = tf.where(a)
z2 = tf.where(a, x, y)

sess = tf.Session()
print(sess.run(z))
print(sess.run(z2))
sess.close()

# z==>[[0]
#      [3]]
# z2==>[ 1 6 7 4]
