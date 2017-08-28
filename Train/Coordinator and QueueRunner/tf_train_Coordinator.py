"""class tf.train.Coordinator

A coordinator for threads.
一个线程的管理者

This class implements a simple mechanism to coordinate the termination of a set of threads.
这个类实现了一个简单的机制来协调一组线程的终止。

Usage(用法)

# Create a coordinator.创建coordinator管理者类
coord = Coordinator()
# Start a number of threads, passing the coordinator to each of them.
# 通过他们对应的协调者开启一组线程
...start thread 1...(coord, ...)
...start thread N...(coord, ...)
# Wait for all the threads to terminate.等待所有线程终止
coord.join(threads)
Any of the threads can call coord.request_stop() to ask for all the threads
to stop. To cooperate with the requests, each thread must check for
coord.should_stop() on a regular basis. coord.should_stop() returns
True as soon as coord.request_stop() has been called.
所有的线程都可以调用coord.request_stop()函数来是所有的线程中止,每一个线程必须定期检查coord.should_stop()函数,
一旦coord.request_stop()函数被调用,coord.should_stop()函数返回True.

A typical thread running with a Coordinator will do something like:
一个典型的利用线程协调器管理的线程会像如下的方式进行工作

while not coord.should_stop():
   ...do some work...
"""