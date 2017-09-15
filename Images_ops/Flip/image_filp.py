"""tensorflow内部含有实现图像翻转的函数为
tf.image.flip_up_down：从上向下翻转
tf.image.flip_left_right：从左到又翻转
tf.image.transpose_image：对角线翻转
tf.image.random_flip_up_down：以一定概率从上向下翻转
tf.image.random_flip_left_right：以一定概率从左到又翻转"""

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('../test_images/test_2.jpg', 'rb').read()

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    img_1 = tf.image.flip_up_down(img_data_jpg)
    img_2 = tf.image.flip_left_right(img_data_jpg)
    img_3 = tf.image.transpose_image(img_data_jpg)

    plt.figure(1)
    plt.imshow(img_1.eval())
    plt.figure(2)
    plt.imshow(img_2.eval())
    plt.figure(3)
    plt.imshow(img_3.eval())
    plt.show()