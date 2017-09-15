"""tensorflow有两种方式：Session.run和 Tensor.eval，这两者的区别在哪？
答：
如果你有一个Tensor t，在使用t.eval()时，等价于：tf.get_default_session().run(t).
举例："""
import tensorflow as tf

t = tf.constant(42.0)
sess = tf.Session()
with sess.as_default():  # or `with sess:` to close on exit
    assert sess is tf.get_default_session()
    assert t.eval() == sess.run(t)
"""这其中最主要的区别就在于你可以使用sess.run()在同一步获取多个tensor中的值，"""
t = tf.constant(42.0)
u = tf.constant(37.0)
tu = tf.multiply(t, u)
ut = tf.multiply(u, t)
with sess.as_default():
    tu.eval()  # runs one step
    ut.eval()  # runs one step
    sess.run([tu, ut])  # evaluates both tensors in a single step
    # 注意到：每次使用 eval 和 run时，都会执行整个计算图，为了获取计算的结果，将它分配给tf.Variable，然后获取。
    # 注意在使用eval函数时需要将sess设置为default Session