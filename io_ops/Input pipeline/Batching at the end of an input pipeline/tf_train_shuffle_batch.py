"""tf.train.shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)

Creates batches by randomly shuffling tensors.
通过随机打乱张量的顺序创建批次.

简单来说就是读取一个文件并且加载一个张量中的batch_size行

This function adds the following to the current Graph:
这个函数将以下内容加入到现有的图中.

A shuffling queue into which tensors from tensor_list are enqueued.
一个由传入张量组成的随机乱序队列
A dequeue_many operation to create batches from the queue.
从张量队列中取出张量的出队操作
A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors
from tensor_list.
一个队列运行器管理出队操作.
If enqueue_many is False, tensor_list is assumed to represent a
single example. An input tensor with shape [x, y, z] will be output
as a tensor with shape [batch_size, x, y, z].
If enqueue_many is True, tensor_list is assumed to represent a
batch of examples, where the first dimension is indexed by example,
and all members of tensor_list should have the same size in the
first dimension. If an input tensor has shape [*, x, y, z], the
output will have shape [batch_size, x, y, z].
'enqueue_many’主要是设置tensor中的数据是否能重复,如果想要实现同一个样本多次出现可以将其设置为:“True”,如果只想要其出现一次,也就是保持数据的唯一性,这时候我们将其设置为默认值:"False"
The capacity argument controls the how long the prefetching is allowed to
grow the queues.
容量控制了预抓取操作对于增加队列长度操作的长度.
For example:

# Creates batches of 32 images and 32 labels.
image_batch, label_batch = tf.train.shuffle_batch(
      [single_image, single_label],
      batch_size=32,
      num_threads=4,
      capacity=50000,
      min_after_dequeue=10000)
Args:

tensor_list: The list of tensors to enqueue.
入队的张量列表
batch_size: The new batch size pulled from the queue.
表示进行一次批处理的tensors数量.
capacity: An integer. The maximum number of elements in the queue.
容量:一个整数,队列中的最大的元素数.
这个参数一定要比min_after_dequeue参数的值大,并且决定了我们可以进行预处理操作元素的最大值.
推荐其值为:
capacity=(min_after_dequeue+(num_threads+a small safety margin∗batchsize)
min_after_dequeue: Minimum number elements in the queue after a
dequeue(出列), used to ensure a level of mixing of elements.
当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.
定义了随机取样的缓冲区大小,此参数越大表示更大级别的混合但是会导致启动更加缓慢,并且会占用更多的内存
num_threads: The number of threads enqueuing tensor_list.
设置num_threads的值大于1,使用多个线程在tensor_list中读取文件,这样保证了同一时刻只在一个文件中进行读取操作(但是读取速度依然优于单线程),而不是之前的同时读取多个文件,这种方案的优点是:
避免了两个不同的线程从同一文件中读取用一个样本
避免了过多的磁盘操作
seed: Seed for the random shuffling within the queue.
打乱tensor队列的随机数种子
enqueue_many: Whether each tensor in tensor_list is a single example.
定义tensor_list中的tensor是否冗余.
shapes: (Optional) The shapes for each example. Defaults to the
inferred shapes for tensor_list.
用于改变读取tensor的形状,默认情况下和直接读取的tensor的形状一致.
name: (Optional) A name for the operations.
Returns:

A list of tensors with the same number and types as tensor_list.
默认返回一个和读取tensor_list数据和类型一个tensor列表."""
