"""

def name_scope(name, default_name=None, values=None):

Wrapper for Graph.name_scope() using the default graph.
使用默认图包装“Graph.name_scope()
See
Graph.name_scope()
for more details.

Args:

name: A name for the scope.
Returns:

A context manager that installs name as a new name scope in the
default graph.
在默认图中安装名称作为一个新名称范围的内容管理器
"""