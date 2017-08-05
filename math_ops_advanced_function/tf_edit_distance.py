import tensorflow as tf

"""tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')
功能：计算Levenshtein距离。
输入：hypothesis:'SparseTensor';
     truth:'SparseTensor'.
     莱文斯坦距离，又称Levenshtein距离，是编辑距离的一种。指两个字串之间，由一个转成另一个所需的最少编辑操作次数。
     允许的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。"""

hypothesis = tf.SparseTensor(
    [[0, 0, 0],
     [1, 0, 0]],
    ["a", "b"],
    (2, 1, 1))
truth = tf.SparseTensor(
    [[0, 1, 0],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0]],
    ["a", "b", "c", "a"],
    (2, 2, 2))
z = tf.edit_distance(hypothesis, truth)

sess = tf.Session()
print(sess.run(z))
sess.close()

# z==>[[inf 1.]
#      [0.5 1.]]
