"""tf.cross(x,y,name=None)
功能：计算叉乘。最大维度为3。
输入：x,y具有相同尺寸的tensor，包含3个元素的向量"""
import tensorflow as tf

x = tf.constant([[1, 2, -3]], tf.float64)
y = tf.constant([[2, 3, 4]], tf.float64)
z = tf.cross(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# z==>[[17. -10. -1]]#2×4-（-3）×3=17，-（1×4-（-3）×2）=-10，1×3-2×2=-1。

"""报错：使用多于三个维度的向量"""
# x = tf.constant([[1, 2, -3, 4]], tf.float64)
# y = tf.constant([[2, 3, 4, 3]], tf.float64)
# z = tf.cross(x, y)
# sess = tf.Session()
# print(sess.run(z))
# sess.close()

"""可以使用二维数组"""
x = tf.constant([[1, 2, -3], [1, 2, -3]], tf.float64)
y = tf.constant([[2, 3, 4], [3, 4, 5]], tf.float64)
z = tf.cross(x, y)
sess = tf.Session()
print(sess.run(z))
sess.close()
# [[ 17. -10.  -1.] [ 22. -14.  -2.]]

"""报错：对应维度必须相同"""
# x = tf.constant([[1, 2]], tf.float64)
# y = tf.constant([[2, 3, 4]], tf.float64)
# z = tf.cross(x, y)
# sess = tf.Session()
# print(sess.run(z))
# sess.close()
"""
叉乘得到的结果是一个标量
a×b=c,其中|c|=|a||b|·sinθ,c的方向遵守右手定则
二维计算方法向量积|c|=|a×b|=|a||b|sin<a,b>
即c的长度在数值上等于以a，b，夹角为θ组成的平行四边形的面积。
三维计算方法
u = Xu*i + Yu*j + Zu*k;
v = Xv*i + Yv*j + Zv*k;
uxv = (Yu*Zv – Zu*Yv)*i+(Zu*Xv – Xu*Zv)*j+(Xu*Yv – Yu*Xv)*k
"""