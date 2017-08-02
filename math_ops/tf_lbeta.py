import tensorflow as tf

"""tf.lbeta(x,name=None)
功能：计算`ln(|Beta(x)|)`,并以最末尺度进行归纳。
          最末尺度`z = [z_0,...,z_{K-1}]`，则Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)
输入：x为秩为n+1的张量，可以为'float','double'类型。"""
x = tf.constant([[4, 3, 3], [2, 3, 2]], tf.float64)
z = tf.lbeta(x)

# ln(gamma(4)*gamma(3)*gamma(3)/gamma(4+3+3))=ln(6*2*2/362880)=-9.62377365
# ln(gamma(2)*gamma(3)*gamma(2)/gamma(2+3+2))=ln(2/720)=-5.88610403
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[-9.62377365 -5.88610403]
# 这是beta函数的计算法，是以gama函数作为基础的所谓伽马函数即是（n-1）!
# 例如：gama(4)=3*2*1=6
