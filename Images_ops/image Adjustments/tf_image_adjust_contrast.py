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

"""调节图片对比度,图片对比度降低则将会生成一个识别度相当差的新图像.
最好选择一个较小的增量,过大的增量会导致图片饱和,像素会呈现全黑或全白的情况"""
# 增加对比度可以更明显的突出颜色的变化.
adjust_contrast = tf.image.adjust_contrast(image, -.5)

print("调节图片的对比度减少0.5", sess.run(tf.slice(adjust_contrast, [1, 0, 0], [1, 3, 3])))
# 调节图片的对比度减少0.5
# [[[169  76 125]
#   [171 126  13]
#   [167  71 122]]]
