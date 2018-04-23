import numpy as np
import tensorflow as tf
import data

tf_idf_data = data.tf_idf_papers_vs_words_freq()

number_words = 8000

def gaussian_additive_noise(x, std):
  return x + tf.random_normal(shape=tf.shape(x), dtype=tf.float32, mean=0.0, stddev=std)

X = tf.placeholder(tf.float32, shape=[None, number_words])

# noise = gaussian_additive_noise(X, 0.1)
# corrupted_X_test = noise.eval(session=tf.Session(), feed_dict={X: tf_idf_data})


def autoencoder(dims=[28*28, 512, 256, 64], std=0.01):
  x = tf.placeholder(tf.float32, shape=[None, dims[0]], name="Input")

  cur = gaussian_additive_noise(x, 0.1)

  Ws = []
  bs = []

  # encoder
  for i, n_out in enumerate(dims[1:]):
    n_inp = int(cur.get_shape()[1])

    W = tf.Variable(tf.random_normal(shape=[n_inp, n_out], mean=0.0, stddev=std, dtype=tf.float32))
    b = tf.Variable(tf.random_normal(shape=[n_out], mean=0.0, stddev=std, dtype=tf.float32))

    Ws.append(W)
    bs.append(b)

    out = tf.nn.tanh(cur @ W + b)
    cur = out

  z = cur
  Ws.reverse()
  bs.reverse()

  # decoder
  for i, n_out in enumerate(dims[:-1][::-1]):
    W = tf.transpose(Ws[i])
    b = tf.Variable(tf.random_normal(shape=[n_out], mean=0.0, stddev=std, dtype=tf.float32))

    out = tf.nn.tanh(cur @ W + b)
    cur = out

  y = cur

  loss = tf.reduce_mean(tf.square(y - x))

  return (x, z, y, loss)


lr = 0.001
batch_size = 64
n_epochs = 50
num_papers = 16980
n_batchs = num_papers // batch_size

x, z, y, loss = autoencoder(dims=[28*28, 512, 256, 64], std=0.01)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


S = tf.Session()
S.run(tf.global_variables_initializer())

num_batches = int(num_papers / batch_size)
batches = np.array_split(tf_idf_data, batch_size)

for i_epoch in range(1, n_epochs+1):
    loss_avg = 0.0

    for i_batch in batches:
        print('i_batch shape ', i_batch.shape)
        _, loss_val = S.run([optimizer, loss], feed_dict={x: i_batch})
        loss_avg = (loss_val / batch_size)

    print(i_epoch, loss_avg)

    loss_avg = 0.0

n_samples = 10
reconstructed = S.run([y], feed_dict={x: X})

reconstructed_0 = reconstructed[0]
print(str(reconstructed_0))