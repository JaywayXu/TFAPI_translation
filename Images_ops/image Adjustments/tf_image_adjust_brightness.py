import tensorflow as tf
# 调节图像光亮
example_red_pixel = tf.constant([254., 2., 15.])
adjust_brightness = tf.image.adjust_brightness(example_red_pixel, 0.2)
# 调整亮度,其实也就是将像素点的所有通道值中的数都变加上delta
sess = tf.Session()
print(sess.run(adjust_brightness))
# [ 254.19999695    2.20000005   15.19999981]
# 返回一个和原来图片shape一样的图片,其中delta数值是一个在[0, 1)的数