"""Session类负责数据流图的执行,构造方法为 def __init__(self, target='', graph=None, config=None):
target指定了所要使用的执行引擎,对于大多数的应用该参数取为默认的空白字符串,
在分布式设置中选择使用Session对象时,该参数用于连接不同的tf.train.Server实例

graph参数指定了将要在Session对象中加载的Graph对象,其默认值为None,表示将使用当前默认的数据流图.
当时用多个数据流图时,最好的方式是显式的传入你希望运行的Graph对象(而非在于一个with语句快中创建Session对象)

config参数允许用户指定配置的Session对象所需的选项,如限制CPU或GPU的使用数目,为数据流图设置优化参数及日志选项
在典型的Tensorflow程序中,创建Session对象无需改变任何默认构造参数"""
import tensorflow as tf
tf.Session()