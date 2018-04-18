import tensorflow as tf
import data

nb_users = 5551  # users
nb_products = 16980  # products
nb_factors = 5000  # no of latent factors
# max_rating = 1	# max possible rating for a paper
# nb_rated_products = 5	# avg rated products per user
top_k_products = 35

uim = data.read_and_create_user_Matrix()
print('shape of uim')
print(uim.shape)
graph = tf.Graph()

with graph.as_default():
  # User-item matrix
  user_item_matrix = tf.placeholder(tf.float32, shape=(nb_users, nb_products))

  # SVD
  St, Ut, Vt = tf.svd(user_item_matrix,  full_matrices=True)
  print('-' * 50)
  print('shape of St')
  print(St.shape)
  print('-' * 50)
  print('shape of Ut')
  print(Ut.shape)
  print('-' * 50)
  print('shape of Vt')
  print(Vt.shape)
  print('-' * 50)

  # Compute reduced matrices
  Sk = tf.diag(St)[0:nb_factors, 0:nb_factors]
  Uk = Ut[:, 0:nb_factors]
  Vk = Vt[0:nb_factors, :]

  print('-' * 50)
  print('shape of Sk')
  print(Sk.shape)
  print('-' * 50)
  print('shape of Uk')
  print(Uk.shape)
  print('-' * 50)
  print('shape of Vk')
  print(Vk.shape)
  print('-' * 50)

  # quit()
  # Compute Su and Si
  Su = tf.matmul(Uk, tf.sqrt(Sk))
  Si = tf.matmul(tf.sqrt(Sk), Vk)

  print('-' * 50)
  print('shape of Su')
  print(Su.shape)
  print('-' * 50)
  print('shape of Si')
  print(Si.shape)
  print('-' * 50)

  # Compute user ratings
  ratings_t = tf.matmul(Su, Si)
  print('-' * 50)
  print('shape of ratings_t')
  print(ratings_t.shape)
  print('-' * 50)
  # Pick top k suggestions
  best_ratings_t, best_items_t = tf.nn.top_k(ratings_t, top_k_products)


  # Create Tensorflow session
session = tf.InteractiveSession(graph=graph)

# Compute the top k suggestions for all users
feed_dict = {
  user_item_matrix: uim
}

best_items = session.run([best_items_t], feed_dict=feed_dict)
# Suggestions for user 10, 20
for i in range(1, 10):
  print('User {}: {}'.format(i, best_items[0][i]))

text_file = open("res_recom.txt", "w")
for i in range(0, nb_users):
  text_file.write('User {}: {}\n'.format(i, best_items[0][i]))
text_file.close()
print(type(best_items))
print('shape of best items')
