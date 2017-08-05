import tensorflow as tf

"""tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)random_gamma
功能：对每一个给定的gamma分布进行shape尺度的采样。
     例如samples = tf.random_gamma([30], [[1.],[3.],[5.]], beta=[[3., 4.]])
      即给定了6个gamma分布，每个分布输出30个数据，输出tensor形状为[30,3,2]。
输入：shape：每一个gamma分布进行采样的尺度;
     alpha： gamma分布的alpha变量，可以为任意尺度，但需和beta对应;
     beta：gamma分布的beta变量。"""

a = tf.random_gamma([2], [0.5, 1.5])  # 每一片应为[:,0],[:,1]

sess = tf.Session()
print(sess.run(a))
sess.close()

# a==>[[ 0.00141039  0.6760782 ]
#       [ 1.04439545  0.37316594]]
# about:http://www.52nlp.cn/lda-math-%E7%A5%9E%E5%A5%87%E7%9A%84gamma%E5%87%BD%E6%95%B03
