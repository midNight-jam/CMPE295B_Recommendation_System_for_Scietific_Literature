import data
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import svd_rec
import config


matrix = data.read_and_create_user_Matrix()
print('-' * 30)
print('SHape of matrix from data.py')
print(matrix.shape)
print('-' * 30)

print('-' * 30)
print(config.right_now)
print('-' * 30)


num_input = config.numbers_papers

num_hidden_5=8192
num_hidden_4=4096
num_hidden_3=2048
num_hidden_2=1024
num_hidden_1=512

X = tf.placeholder(tf.float64, [None, num_input])

print(X.shape)
print('-' * 30)

# quit()

weights = {
    'encoder_h5': tf.Variable(tf.random_normal([num_input, num_hidden_5], dtype=tf.float64)),
    'encoder_h4': tf.Variable(tf.random_normal([num_hidden_5, num_hidden_4], dtype=tf.float64)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_3], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2], dtype=tf.float64)),
    'encoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3], dtype=tf.float64)),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4], dtype=tf.float64)),
    'decoder_h4': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_5], dtype=tf.float64))
    'decoder_h5': tf.Variable(tf.random_normal([num_hidden_5, num_input], dtype=tf.float64))
}

biases = {
    'encoder_b5': tf.Variable(tf.random_normal([num_hidden_5], dtype=tf.float64)),
    'encoder_b4': tf.Variable(tf.random_normal([num_hidden_4], dtype=tf.float64)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_3], dtype=tf.float64)),
    'decoder_b3': tf.Variable(tf.random_normal([num_hidden_4], dtype=tf.float64)),
    'decoder_b4': tf.Variable(tf.random_normal([num_hidden_5], dtype=tf.float64))
    'decoder_b5': tf.Variable(tf.random_normal([num_input], dtype=tf.float64))
}


def encoder(x):
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h5']), biases['encoder_b5']))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h4']), biases['encoder_b4']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_4, weights['encoder_h3']), biases['encoder_b3']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['encoder_h2']), biases['encoder_b2']))
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h1']), biases['encoder_b1']))
    return layer_1


def decoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']), biases['decoder_b5']))
    return layer_5


# Construct models
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


# Prediction
y_pred = decoder_op


# Targets are the input data.
y_true = X


# Define loss and optimizer, minimize the squared error
# loss = tf.losses.mean_squared_error(y_true, y_pred)
loss = tf.losses.sigmoid_cross_entropy(y_true, y_pred)


# optimizer = tf.train.RMSPropOptimizer(0.003).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(loss)
# optimizer = tf.train.AdagradOptimizer(1e-4).minimize(loss)


predictions = pd.DataFrame()

# Define evaluation metrics

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

losses_plot = []

prev_loss = 0.0

print('running....')
with tf.Session() as session:
    epochs = config.epoch
    batch_size = config.batch_size

    session.run(init)
    session.run(local_init)

    num_batches = int(matrix.shape[0] / batch_size)
    batches = np.array_split(matrix, batch_size)

    for i in range(epochs):

        avg_cost = 0

        for batch in batches:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        prev_loss = avg_cost
        avg_cost /= num_batches

        print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

        losses_plot.append(avg_cost)

    print("Predictions...")

    matrix = np.concatenate(batches, axis=0)

    preds = session.run(decoder_op, feed_dict={X: matrix})
    
    
    predictions = predictions.append(pd.DataFrame(preds))

    print('Generated Predictions Shape After Atuoencoders: ',predictions.shape)


    # now doing the matrix factorization

    svd_rec.process(predictions)

    predictions.to_csv(config.autoencoders_output_file, header=False, index=False)
    print('to csv complete')

################################################
## Persist the results
################################################