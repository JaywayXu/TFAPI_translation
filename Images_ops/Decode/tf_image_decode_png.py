"""tf.image.decode_png(contents, channels=None, name=None)

Decode a PNG-encoded image to a uint8 tensor.
将一个png编码的图像解码成一个uint8张量。
The attr channels indicates the desired number of color channels for the
decoded image.
参数"channels"表示解码图像所需的颜色通道数量。

Accepted values are:

0: Use the number of channels in the PNG-encoded image.
使用png编码图像的通道数
1: output a grayscale image.
输出一个灰度图像
3: output an RGB image.
输出一个RGB值表示的图像
4: output an RGBA image.
输出一个RGBA值表示的图像
If needed, the PNG-encoded image is transformed to match the requested number
of color channels.
如果需要，将转换为png编码的图像，以匹配所请求的颜色通道数量。
Args:

contents: A Tensor of type string. 0-D. The PNG-encoded image.
0阶"string"类型的张量,使用png编码格式的图片
channels: An optional int. Defaults to 0.
Number of color channels for the decoded image.
可选int参数,默认是0,表示解码图像的颜色通道数。
name: A name for the operation (optional).
Returns:

A Tensor of type uint8. 3-D with shape [height, width, channels].
一个uint8类型的张量。3维分别表示[高度，宽度，通道数]
"""