import tensorflow as tf
# Initialize the matrix factors from random normals with mean 0. W will
# represent users and H will represent items.
num_users = 10 # users
num_items = 20 # products
rank = 5 # no of latent factors
W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.2, mean=0), name="users")
H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.2, mean=0), name="items")

# To the user matrix we add a bias column holding the bias of each user,
# and another column of 1s to multiply the item bias by.
# W_plus_bias = tf.concat(1, [W, tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((num_users,1), dtype=float32, name="item_bias_ones")])
# To the item matrix we add a row of 1s to multiply the user bias by, and
# a bias row holding the bias of each item.
# H_plus_bias = tf.concat(0, [H, tf.ones((1, num_items), name="user_bias_ones", dtype=float32), tf.convert_to_tensor(item_bias, dtype=float32, name="item_bias")])
# Multiply the factors to get our result as a dense matrix
# result = tf.matmul(W_plus_bias, H_plus_bias)
result = tf.matmul(W, H)

# Now we just want the values represented by the pairs of user and item
# indices for which we had known ratings. Unfortunately TensorFlow does not
# yet support numpy-like indexing of tensors. See the issue for this at
# https://github.com/tensorflow/tensorflow/issues/206 The workaround here
# came from https://github.com/tensorflow/tensorflow/issues/418 and is a
# little esoteric but in numpy this would just be done as follows:
# result_values = result[user_indices, item_indices]
# result_values = tf.gather(tf.reshape(result, [-1]), user_indices * tf.shape(result)[1] + item_indices, name="extract_training_ratings")

# Same thing for the validation set ratings.
# result_values_val = tf.gather(tf.reshape(result, [-1]), user_indices_val * tf.shape(result)[1] + item_indices_val, name="extract_validation_ratings")

# Calculate the difference between the predicted ratings and the actual
# ratings. The predicted ratings are the values obtained form the matrix
# multiplication with the mean rating added on.
# diff_op = tf.sub(tf.add(result_values, mean_rating, name="add_mean"), rating_values, name="raw_training_error")
# diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val, name="raw_validation_error")

# with tf.name_scope("training_cost") as scope:
    # base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")
    # Add regularization.
    # regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    # cost = tf.div(tf.add(base_cost, regularizer), num_ratings * 2, name="average_error")

# with tf.name_scope("validation_cost") as scope:
    # cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), num_ratings_val * 2, name="average_error")

# Use an exponentially decaying learning rate.
global_step = tf.Variable(0, trainable=False)
lr= 0.1
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Passing global_step to minimize() will increment it at each step so
    # that the learning rate will be decayed at the specified intervals.
    train_step = optimizer.minimize(cost, global_step=global_step)

with tf.name_scope("training_accuracy") as scope:
  # Just measure the absolute difference against the threshold
  # TODO: support percentage-based thresholds
  good = tf.less(tf.abs(diff_op), threshold)

  accuracy_tr = tf.div(tf.reduce_sum(tf.cast(good, tf.float32)), num_ratings)
  accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)

with tf.name_scope("validation_accuracy") as scope:
  # Validation set accuracy:
  good_val = tf.less(tf.abs(diff_op_val), threshold)
  accuracy_val = tf.reduce_sum(tf.cast(good_val, tf.float32)) / num_ratings_val
  accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)

# Create a TensorFlow session and initialize variables.
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Make sure summaries get written to the logs.
summary_op = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph_def)

# Run the graph and see how we're doing on every 500th iteration.
for i in range(max_iter):
    if i % 500 == 0:
        res = sess.run([summary_op, accuracy_tr, accuracy_val, cost, cost_val])
        summary_str = res[0]
        acc_tr = res[1]
        acc_val = res[2]
        cost_ev = res[3]
        cost_val_ev = res[4]
        writer.add_summary(summary_str, i)
        print("Training accuracy at step %s: %s" % (i, acc_tr))
        print("Validation accuracy at step %s: %s" % (i, acc_val))
        print("Training cost: %s" % (cost_ev))
        print("Validation cost: %s" % (cost_val_ev))
    else:
        sess.run(train_step)

with tf.name_scope("final_model") as scope:
    # At the end we want to get the final ratings matrix by adding the mean
    # to the result matrix and doing any further processing required
    add_mean_final = tf.add(result, mean_rating, name="add_mean_final")
    if result_processor == None:
        final_matrix = add_mean_final
    else:
        final_matrix = result_processor(add_mean_final)
    final_res = sess.run([final_matrix])