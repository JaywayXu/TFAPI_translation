import tensorflow as tf

"""tf.matrix_transpose(a,name='matrix_transpose')
功能：进行矩阵转置。只对低维度的2维矩阵转置，功能同tf.transpose(a,perm=[0,1,3,2])。(若a为4维)"""