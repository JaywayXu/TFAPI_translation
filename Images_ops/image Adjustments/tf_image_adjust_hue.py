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
"""改变图片的色度,使色彩更加丰富,该调整函数接受一个delta参数,用于控制需要调节的色度数量"""
adjust_hue = tf.image.adjust_hue(image, 0.7)

print("调节图片色度增加0.7", sess.run(tf.slice(adjust_hue, [1, 0, 0], [1, 3, 3])))
# 调节图片色度增加0.7
# [[[195  38   5]
#   [ 49 227   6]
#   [205  46  11]]]