import tensorflow as tf

"""tf.invert_permutation(x, name=None)
计算张量的逆排序
功能：转换坐标与值。`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`  (for i in range (lens(x))。"""

a = tf.constant([3, 4, 0, 2, 1])
# b = tf.constant([4, 5, 7, 9])
z = tf.invert_permutation(a)
# z1 = tf.invert_permutation(b)
sess = tf.Session()
print(sess.run(z))
# print(sess.run(z1))  # 必须满足索引的条件
sess.close()
# z==>[2 4 3 0 1]
"""此处x=[3, 4, 0, 2, 1]
所以y[3]=0,y[4]=1,y[0]=2,y[2]=3,y[3]=4"""
