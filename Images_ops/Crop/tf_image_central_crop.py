"""在大多数场景中,对图像的操作最好能在预处理阶段完成.预处理包括对图像裁剪,缩放以及灰度调整.
另一方面,在训练时对图像进行操作有一个重要的用例.当一副图像被加载后,可对其进行翻转或扭曲处理,
以使输入给网络的训练信息多样化.虽然这个步骤会进一步增加处理时间,但却有助于缓解过拟合现象"""
"""
def central_crop(image, central_fraction):
  Crop the central region of the image.
  裁剪图像的中心区域.
  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify central_fraction = 0.5, this function
  returns the region marked with "X" in the below diagram.
  删除的外层部分图像但保留图像的中心区域以及每个维度。
  如果我们指定central_fraction = 0.5,这个函数返回该地区标有“X”下面的图。
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------

  Args:
    image: 3-D float Tensor of shape [height, width, depth]
    image: 3维float张量[高度,宽度,深度]深度就是指的通道数,具体也可以看做是图像的显示形式,例如图像的RGB值
    central_fraction: float (0, 1], fraction of size to crop
    central_fraction: float(0, 1],裁剪的倍数
  Raises:
    ValueError: if central_crop_fraction is not within (0, 1].

  Returns:
    3-D float Tensor
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
print("从图像中心将10%抠出", sess.run(tf.image.central_crop(image, 0.1)))
# the image is:
#  [[[  0   9   4]
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
# 从图像中心将10%抠出 [[[  6  94 227]]]
