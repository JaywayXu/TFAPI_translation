"""
tf.map_fn(fn, elems)：接受一个函数对象，然后用该函数对象对集合（elems）中的每一个元素分别处理，

def preprocessing_image(image, training):
    image = ...
    return image

def preprocessing_images(images, training):
    images = tf.map_fn(lambda image: preprocessing_image(image, training), images)
    return images
"""