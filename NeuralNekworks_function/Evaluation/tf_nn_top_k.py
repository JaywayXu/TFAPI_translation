"""
tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，
返回一个bool类型的张量，tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，
大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。target就是实际样本类别的标签，大小就是样本数量的个数。
K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。
"""
# import tensorflow as tf
#
# A = [[0.8, 0.6, 0.3], [0.1, 0.6, 0.4]]
# B = [1, 1]
# out = tf.nn.in_top_k(A, B, 1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(out))
    # [False  True]
"""
解释：因为A张量里面的第一个元素的最大值的标签是0，第二个元素的最大值的标签是1.也就是说
A张量中最大值的坐标是0(0.8),第二个元素最大值的坐标是1(0.6)
但是实际的确是1和1.所以输出就是False 和True。如果把K改成2，那么第一个元素的前面2个最大的元素的位置是0，1，第二个的就是1，2。实际结果是1和1。
包含在里面，所以输出结果就是True 和True.如果K的值大于张量A的列，那就表示输出结果都是true
"""
