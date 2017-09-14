"""tf.image.decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None, try_recover_truncated=None, acceptable_fraction=None, name=None)

Decode a JPEG-encoded image to a uint8 tensor.
讲一个jepg编码的图片编码为int类型的张量
The attr channels indicates the desired number of color channels for the
decoded image.
属"channels"表示解码图像所需的颜色通道数量。
Accepted values are:

0: Use the number of channels in the JPEG-encoded image.
1: output a grayscale image.
3: output an RGB image.
0: 使用jpeg编码映像中的通道数量。
1: 输出一个灰度图像。
3: 输出一个RGB图像
If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.
如果需要，则将jpeg编码的图像转换为匹配请求的数字的颜色通道。
The attr ratio allows downscaling the image by an integer factor during
decoding. Allowed values are: 1, 2, 4, and 8. This is much faster than
downscaling the image later.
在解码过程中，属性’ratio’允许通过整数因子来缩小图像。允许的值是:1、2、4和8。
这比稍后缩小图像的速度要快得多。
Args:

contents: A Tensor of type string. 0-D. The JPEG-encoded image.
是一个0维度的string类型的使用JRPG编码的张量.
channels: An optional int. Defaults to 0.
Number of color channels for the decoded image.
一个可选的int类型的参数,默认是0,表示解码图片的通道数.
ratio: An optional int. Defaults to 1. Downscaling ratio.
默认是1,可以通过调节来缩小图片.
fancy_upscaling: An optional bool. Defaults to True.
If true use a slower but nicer upscaling of the
chroma planes (yuv420/422 only).
可选的布尔值.
默认是"True"如果选择true会使用一个慢但是更好的对于chroma层的加速
plane一般是以luma plane、chroma plane的形式出现，其实就是luma层和chroma层，就像RGB，要用三个plane来存。
try_recover_truncated: An optional bool. Defaults to False.
If true try to recover an image from truncated input.
一个可选的布尔值,默认是假,如果选择"true"则会试图从截断的输入中恢复图像。
acceptable_fraction: An optional float. Defaults to 1.
The minimum required fraction of lines before a truncated
input is accepted.
可选的"float"类型的变量,默认为“1”。在被截断的输入被接受之前，最少需要的行数。
name: A name for the operation (optional).
Returns:

A Tensor of type uint8. 3-D with shape [height, width, channels]…
一个uint8类型的张量。3维分别表示[高度，宽度，通道数]

"""
