"""tf.train.Coordinator.request_stop(ex=None)

Request that the threads stop.
# 请求线程中止

After this is called, calls to should_stop() will return True.
在此函数被调用后,should_stop()函数会返回True.然后进程终止.
Args:

ex: .Optional Exception, or Python ‘exc_info’ tuple as returned by
sys.exc_info(). If this is the first call to request_stop() the
corresponding exception is recorded and re-raised from join()"""