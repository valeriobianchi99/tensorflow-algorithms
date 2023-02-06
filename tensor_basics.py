import tensorflow as tf

print(tf.version)

rank0 = tf.Variable("hi", tf.string)
rank1 = tf.Variable(["hello"], tf.string)
rank2 = tf.Variable([[1, 2], [3, 4], [5, 6]], tf.int64)

# print(rank2.shape)

tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 will figure out what value set based on the tensor2 shape (2*3*1)
# print(tensor3)

t = tf.zeros([5, 5, 5, 5])
print(t)
t2 = tf.reshape(t, [125, -1])
print(t2)
