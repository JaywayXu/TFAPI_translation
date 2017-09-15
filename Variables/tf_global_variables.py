# tf.global_variables或者tf.all_variables都是获取程序中的变量，不同的版本是不同的，1.0+版本的tensorflow版本应该用前面的那个，返回的值是变量的一个列表
import tensorflow as tf

v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')

variables = tf.global_variables()

print(variables[0].name)
print(variables[0].value)
print(variables[1].name)
print(variables[1].value)
# v:0
# <bound method Variable.value of <tf.Variable 'v:0' shape=(1,) dtype=float32_ref>>
# v1:0
# <bound method Variable.value of <tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>>