import data
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

# matrix = data.read_and_create_user_Matrix()
matrix = data.read_and_create_trimmed_user_Matrix()
print('*' * 9)
print('SHape of matrix from data.py')
print(matrix.shape)
print('*' * 9)
print(["%d" % x for x in matrix[0]])
print('*' * 29)


# num_input = 16980
num_input = 4500
num_hidden_0=4096
num_hidden_1=2048
num_hidden_2=1024
num_hidden_3=512

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h0': tf.Variable(tf.random_normal([num_input, num_hidden_0], dtype=tf.float64)),
    'encoder_h1': tf.Variable(tf.random_normal([num_hidden_0, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3], dtype=tf.float64)),
    'decoder_h0': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_0], dtype=tf.float64)),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_0, num_input], dtype=tf.float64))
}

biases = {
    'encoder_b0': tf.Variable(tf.random_normal([num_hidden_0], dtype=tf.float64)),
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3], dtype=tf.float64)),
    'decoder_b0': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_0], dtype=tf.float64)),
    'decoder_b3': tf.Variable(tf.random_normal([num_input], dtype=tf.float64))
}


def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_0 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h0']), biases['encoder_b0']))
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(layer_0, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3


# Building the decoder

def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_0 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h0']), biases['decoder_b0']))

    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(layer_0, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3

# Construct model

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

with tf.Session() as session:
    epochs = 10
    batch_size = 100

    session.run(init)
    session.run(local_init)

    num_batches = int(matrix.shape[0] / batch_size)
    matrix = np.array_split(matrix, batch_size)

    for i in range(epochs):

        avg_cost = 0

        for batch in matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l


        avg_cost /= num_batches

        print("Epoch: {} Loss: {}".format(i + 1, avg_cost))
        losses_plot.append(avg_cost);

    print("Predictions...")

    matrix = np.concatenate(matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: matrix})
    
    preds[preds<=0.02]=0.0
    # preds = preds.astype('<H')
    
    predictions = predictions.append(pd.DataFrame(preds))

    print(predictions)
    right_now = str(datetime.datetime.now().time())
    predictions.to_csv('Data/out'+right_now+'.csv', header=False, index=False)


     right_now = str(datetime.datetime.now().isoformat())
     loss_name = 'loss_plot__'+right_now+'.dat'
     loss_file = open(loss_name,"w")
     for lp in losses_plot:
        loss_file.write('{0}\n'.format((lp))
     loss_file.close()