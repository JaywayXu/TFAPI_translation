import tensorflow as tf

"""tf.set_random_seed(seed)
功能：设置随机数种子。
     为确保每次随机数生成数据一致，可以设置随机数种子。随机种子有两种设置方法：
         1、op级别的设置，如前文提高的随机输入tf.random_shuffle(value, seed=None, name=None)中，
          函数变量seed即设置种子值。
         2、graph级别，即tf.set_random_seed(seed)函数，可使整个graph的随机数产生从设置种子中获取。"""
# 未设置种子
a = tf.random_uniform([1])
b = tf.random_normal([1])
print("Session 1")
with tf.Session() as sess1:
    tf.global_variables_initializer().run()
    print(sess1.run(a))
    print(sess1.run(a))
    print(sess1.run(b))
    print(sess1.run(b))
    print("Session 2")
with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    print(sess2.run(a))
    print(sess2.run(a))
    print(sess2.run(b))
    print(sess2.run(b))
    # 运行结果为：
    #   Session 1
    #   [ 0.71059418]
    #   [ 0.71678996]
    #   [ 0.27808592]
    #   [ 0.77504641]
    #   Session 2
    #   [ 0.45193291]
    #   [ 0.74479854]
    #   [-0.01035937]
    #   [ 0.54787332]
    #   因没有设置随机种子，每次运行结果都不一样
    # 2、设置op级别种子
a = tf.random_uniform([1], seed=1)
b = tf.random_normal([1])
print("Session 1")
with tf.Session() as sess1:
    tf.global_variables_initializer().run()
    print(sess1.run(a))
    print(sess1.run(a))
    print(sess1.run(b))
    print(sess1.run(b))
print("Session 2")
with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    print(sess2.run(a))
    print(sess2.run(a))
    print(sess2.run(b))
    print(sess2.run(b))
    # 运行结果为：
    #   Session 1
    #   [ 0.23903739]
    #   [ 0.22267115]
    #   [-0.48983803]
    #   [-0.13116723]
    #   Session 2
    #   [ 0.23903739]
    #   [ 0.22267115]
    #   [-1.77008951]
    #   [-0.18568291]
    #   变量a设置为种子1，每次运行按照种子进行取数，每个Session都从种子的第一个数开始取值。
    # 3、设置graph级别种子
tf.set_random_seed(1)
a = tf.random_uniform([1])
b = tf.random_normal([1])
print("Session 1")
with tf.Session() as sess1:
    tf.global_variables_initializer().run()
    print(sess1.run(a))
    print(sess1.run(a))
    print(sess1.run(b))
    print(sess1.run(b))
print("Session 2")
with tf.Session() as sess2:
    tf.global_variables_initializer().run()
    print(sess2.run(a))
    print(sess2.run(a))
    print(sess2.run(b))
    print(sess2.run(b))
# 运行结果为：
# Session
# 1
# [0.77878559]
# [0.0978868]
# [-0.4487586]
# [-0.82540691]
# Session
# 2
# [0.77878559]
# [0.0978868]
# [-0.4487586]
# [-0.82540691]
# 设置graph级种子后，两次运行结果完全一致。
