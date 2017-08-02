import tensorflow as tf

"""tf.transpose(a,perm=None,name='transpose')
功能：矩阵转置。
输入：tensor，perm代表转置后的维度排列，决定了转置方法，默认为[n-1,....,0]，n为a的维度"""

a = tf.constant([[1, 2, 3], [4, 5, 6]])
z = tf.transpose(a)  # perm为[1,0]，即0维和1维互换。

sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[1 4]
#      [2 5]
#      [3 6]]
