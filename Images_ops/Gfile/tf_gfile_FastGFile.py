"""
tensorflow里面给出了一个函数用来读取图像，不过得到的结果是最原始的图像，
是没有经过解码的图像，这个函数为tf.gfile.FastGFile（‘path’， ‘r’）.read()。
如果要显示读入的图像，那就需要经过解码过程，tensorflow里面提供解码的函数有两个，
tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，得到图像的像素值，这个像素值可以用于显示图像。
如果没有解码，读取的图像是一个字符串，没法显示。
"""
# read方法
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data_jpg = tf.gfile.FastGFile('../test_images/test_1.jpg', 'rb').read()
image_raw_data_png = tf.gfile.FastGFile('../test_images/test_3.png', 'rb').read()

sess = tf.Session()
img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)  # 图像解码
img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)  # 改变图像数据的类型

img_data_png = tf.image.decode_png(image_raw_data_png)
img_data_png = tf.image.convert_image_dtype(img_data_png, dtype=tf.uint8)

plt.figure(1)  # 图像显示
plt.imshow(img_data_jpg.eval(session=sess))  # 对于这个函数相当于sess.run(img_data_jpg)
plt.figure(2)
plt.imshow(img_data_png.eval(session=sess))
plt.show()

"""实现图片的编码和保存"""
# write方法,将图片转换编码格式后保存
image_raw_jpg = tf.gfile.FastGFile('../test_images/test_2.jpg', 'rb').read()

img_data_jpg = tf.image.decode_jpeg(image_raw_jpg)  # 解码
img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)  # 编码为无符号的int8类型
encode_image_jpg = tf.image.encode_jpeg(img_data_jpg)  # jpg编码
encode_image_png = tf.image.encode_png(img_data_jpg)  # png编码

with tf.gfile.GFile('../test_images/output.jpg', 'wb') as f:
    f.write(sess.run(encode_image_jpg))

with tf.gfile.GFile('../test_images/output.png', 'wb') as f:
    f.write(sess.run(encode_image_png))
