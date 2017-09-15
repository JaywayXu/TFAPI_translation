"""tensorflow里面用于改变图像大小的函数是tf.image.resize_images(image， （w， h）， method)：image表示需要改变此存的图像
，第二个参数改变之后图像的大小，method用于表示改变图像过程用的差值方法。0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法。"""
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('../test_images/test_2.jpg', 'rb').read()

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    resize_0 = tf.image.resize_images(img_data_jpg, (500, 500), method=0)
    resize_1 = tf.image.resize_images(img_data_jpg, (500, 500), method=1)
    resize_2 = tf.image.resize_images(img_data_jpg, (500, 500), method=2)
    resize_3 = tf.image.resize_images(img_data_jpg, (500, 500), method=3)

    print(sess.run(tf.shape(resize_0)))
    # [500 500   3]

    plt.figure(0)
    plt.imshow(resize_0.eval())
    plt.figure(1)
    plt.imshow(resize_1.eval())
    plt.figure(2)
    plt.imshow(resize_2.eval())
    plt.figure(3)
    plt.imshow(resize_3.eval())

    plt.show()
