"""边界填充
为使输入图像符合期望的尺寸,可用0进行边界填充"""

"""def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
   Pad `image` with zeros to the specified `height` and `width`.
  
    Adds `offset_height` rows of zeros on top, `offset_width` columns of
    zeros on the left, and then pads the image on the bottom and right
    with zeros until it has dimensions `target_height`, `target_width`.
  
    This op does nothing if `offset_*` is zero and the image already has size
    `target_height` by `target_width`.
  
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or
             3-D Tensor of shape `[height, width, channels]`.
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
  
    Returns:
      If `image` was 4-D, a 4-D float Tensor of shape
      `[batch, target_height, target_width, channels]`
      If `image` was 3-D, a 3-D float Tensor of shape
      `[target_height, target_width, channels]`
  
    Raises:
      ValueError: If the shape of `image` is incompatible with the `offset_*` or
        `target_*` arguments, or either `offset_height` or `offset_width` is
        negative.
    """

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
# 该边界填充方法仅可接受实值输入,所以在进行边界填充操作时,需要将图片先运行
real_image = sess.run(image)

pad = tf.image.pad_to_bounding_box(
    real_image, offset_height=0, offset_width=0, target_height=4, target_width=4)

print("The padding of the image is:", sess.run(pad))
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
# The padding of the image is:
# [[[  0   9   4]
#   [254 255 250]
#   [255  11   8]
#   [  0   0   0]]
#
#  [[ 10 195   5]
#   [  6  94 227]
#   [ 14 205  11]
#   [  0   0   0]]
#
#  [[255  10   4]
#   [249 255 244]
#   [  3   8  11]
#   [  0   0   0]]
#
#  [[  0   0   0]
#   [  0   0   0]
#   [  0   0   0]
#   [  0   0   0]]]
#
# Process finished with exit code 0
