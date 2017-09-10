"""裁剪通常在预处理阶段使用,但在训练阶段,若背景也有用时,可随机化裁剪区域起始位置到图像中心的偏移量来实现裁剪
def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
此函数用于截取从坐标点(offset_height,offset_width)开始高为target_height,宽为target_width的矩形
其中设置图像最左上角开始的点为坐标点(0, 0)
  image: 4-D Tensor of shape `[batch, height, width, channels]` or
         3-D Tensor of shape `[height, width, channels]`.
  Returns:
  If `image` was 4-D, a 4-D float Tensor of shape
  `[batch, target_height, target_width, channels]`
  If `image` was 3-D, a 3-D float Tensor of shape
  `[target_height, target_width, channels]`

"""

# 这个裁剪方式进接受实值输入,即其仅能接受一个具有确定形状的张量,因此,输入图像需要事先在数据流图中运行.

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
real_image = sess.run(image)

bounding_crop = tf.image.crop_to_bounding_box(
    real_image, offset_height=0, offset_width=0, target_height=2, target_width=1)

print("The bounding_crop image is", sess.run(bounding_crop))
#
# the image is:
# [[[  0   9   4]
#   [254 255 250]
#   [255  11   8]]
#
#  [[ 10 195   5]
#   [  6  94 227]
#   [ 14 205  11]]
#
#  [[255  10   4]
#   [249 255 244]
#   [  3   8  11]]]
# The bounding_crop image is
# [[[  0   9   4]]
#
#  [[ 10 195   5]]]

# 这段代码意为从位于(0,0)的图像的左上角像素开始对图像裁剪.