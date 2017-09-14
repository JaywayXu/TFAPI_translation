"""tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量"""
import tensorflow as tf;

with tf.variable_scope('V1'):
    a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
with tf.variable_scope('V2'):
    a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(a4.name)
    # V1/a1: 0
    # V1/a2: 0
    # V2/a1: 0
    # V2/a2: 0

#
# with tf.name_scope('V1'):
#     a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#     a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
# with tf.name_scope('V2'):
#     a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#     a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(a1.name)
#     print(a2.name)
#     print(a3.name)
#     print(a4.name)
"""ValueError: Variable a1 already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:"""

with tf.name_scope('V1'):
    # a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
with tf.name_scope('V2'):
    # a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print a1.name
    print(a2.name)
    # print a3.name
    print(a4.name)
    # V1_1/a2: 0
    # V2_1/a2: 0

"""reuse为True的时候,表示用tf.get_variable得到的变量可以在别的地方重复使用"""
import tensorflow as tf

with tf.variable_scope('V1'):
    b1 = tf.get_variable(name='b1', shape=[1], initializer=tf.global_variables_initializer())

with tf.variable_scope('V1', reuse=True):
    b3 = tf.get_variable('b1')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(b1.name)
    print(sess.run(b1))
    print(b3.name)
    print(sess.run(b3))
    # 分析：变量a1和a3一样的变量，名字和值都是一样的。

# V1/b1:0
# [ 1.]
# V1/b1:0
# [ 1.]
