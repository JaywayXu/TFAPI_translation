# 复用image图像.
import tensorflow as tf

image_filename = "../test_images/test-input-image.jpg"
# 获得文件名列表
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))
# 生成文件名队列
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
# 通过阅读器返回一个键值对,其中value表示图像
image = tf.image.decode_jpeg(image_file)
# 通过tf.image.decode_jpeg解码函数对图片进行解码,得到图像.
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
print('the image is:', sess.run(image))
filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
"""对于训练集中的图像存在多种不同的长宽比,tensorflow提供了一种组合pad和crop的尺寸调整方法"""
# 将图片调整为高为2,宽为5的矩形图片
# real_imag高度被减小到两个像素,边界通过填充0像素使宽度得以增加.
real_image = sess.run(image)
crop_or_pad = tf.image.resize_image_with_crop_or_pad(
    real_image, target_height=2, target_width=5)

print("the crop_and_pad of the image is", sess.run(crop_or_pad))
# the crop_and_pad of the image is
# [[[  0   0   0]
#   [  0   9   4]
#   [254 255 250]
#   [255  11   8]
#   [  0   0   0]]
#
#  [[  0   0   0]
#   [ 10 195   5]
#   [  6  94 227]
#   [ 14 205  11]
#   [  0   0   0]]]