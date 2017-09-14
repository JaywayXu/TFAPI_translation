"""tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')

Starts all queue runners collected in the graph.
运行所有在图中的队列运行器.

This is a companion method to add_queue_runner(). It just starts
threads for all queue runners collected in the graph. It returns
the list of all threads.
这是"add_queue_runner()"函数的配合用法,运行在图中存储的所有队列运行器,返回所有的线程的列表

Args:

sess: Session used to run the queue ops. Defaults to the
default session.
coord: Optional Coordinator for coordinating the started threads.
daemon: Whether the threads should be marked as daemons(守护进程), meaning
they don’t block program exit.(表示他们不阻断程序的退出)
start: Set to False to only create the threads, not start them.
如果设置为"False"表示只创建线程但是不开始他们
collection: A GraphKey specifying the graph collection to
get the queue runners from. Defaults to GraphKeys.QUEUE_RUNNERS.
指定图集合获取队列运行器.
Returns:

A list of threads.
一系列线程."""