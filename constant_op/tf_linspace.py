import tensorflow as tf

"""tf.linspace(start, stop, num, name=None)
功能：生成在区间[start,stop]中定长间隔的值。序列值的间隔大小为‘(stop-start)/(num-1)’
输入：start：区间起始值，类型为float32或float64;
     stop：区间中止值，类型为float32或float64;
     num：生成数据数量。"""

a = tf.linspace(1., 7., 4)

sess = tf.Session()
print(sess.run(a))
sess.close()
# a==>[1. 3. 5. 7.]
