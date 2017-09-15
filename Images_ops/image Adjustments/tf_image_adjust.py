"""实现图像的调整包括以下几个方面：
调整图像的亮度：tf.image.adjust_brightness（img， p）。img是表示目标图像，p表示对比度，大于零变亮，反之变暗
调整图像的对比度。tf.image.adjust_contrast，用法同上
调整色相。tf.image.adjust_hue。用法同上
调整饱和度：tf.image.adjust_saturation。用法同上"""
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('../test_images/test_2.jpg', 'rb').read()

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    img_1 = tf.image.adjust_brightness(img_data_jpg, 0.04)
    img_2 = tf.image.adjust_contrast(img_data_jpg, -0.3)
    img_3 = tf.image.adjust_hue(img_data_jpg, 0.8)
    img_4 = tf.image.adjust_saturation(img_data_jpg, -5)
    plt.figure(1)
    plt.imshow(img_1.eval())
    plt.figure(2)
    plt.imshow(img_2.eval())
    plt.figure(3)
    plt.imshow(img_3.eval())
    plt.figure(4)
    plt.imshow(img_4.eval())
    plt.show()
