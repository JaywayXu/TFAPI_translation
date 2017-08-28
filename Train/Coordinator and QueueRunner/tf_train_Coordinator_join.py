"""tf.train.Coordinator.join(threads, stop_grace_period_secs=120)

Wait for threads to terminate.
等待线程终止
Blocks until all ‘threads’ have terminated or request_stop() is called.
直到所有的“线程”都被终止或请求停止()被调用。
After the threads stop, if an ‘exc_info’ was passed to request_stop, that
exception is re-reaised.线程停止后，如果“exc_info”异常传递给request_stop，则会有重新启动线程的异常.

Grace period handling: When request_stop() is called, threads are given
’stop_grace_period_secs’ seconds to terminate. If any of them is still
alive after that period expires, a RuntimeError is raised. Note that if
an ‘exc_info’ was passed to request_stop() then it is raised instead of
that RuntimeError.
宽限期处理:当request_stop()函数被调用时,线程会被给予’stop_grace_period_secs’秒时间处理暂停,如果这个期间过后仍然有线程在执行,运行时的错误就会发生.请注意，如果将“excinfo”错误传递给requeststop()，那么它将会被先提出，而不是运行时错误。

Args:

threads: List threading.Threads. The started threads to join.
线程列表
stop_grace_period_secs: Number of seconds given to threads to stop after
request_stop() has been called.
当request_stop函数被调用后继续等待的时间.
Raises:

RuntimeError: If any thread is still alive after request_stop()
is called and the grace period expires.
在request_stop被调用后并且过了宽限期,要是线程还在执行."""