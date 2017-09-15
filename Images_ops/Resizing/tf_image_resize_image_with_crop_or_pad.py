"""tensorflow里面提供了实现图像进行裁剪和填充的函数，
就是tf.image.resize_image_with_crop_or_pad（img，height，width ）。
img表示需要改变的图像，height是改变后图像的高度，width是宽度。"""
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('../test_images/test_2.jpg', 'rb').read()

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    crop = tf.image.resize_image_with_crop_or_pad(img_data_jpg, 500, 500)
    pad = tf.image.resize_image_with_crop_or_pad(img_data_jpg, 2000, 2000)

    plt.figure(1)
    plt.imshow(crop.eval())
    plt.figure(2)
    plt.imshow(pad.eval())
    plt.show()