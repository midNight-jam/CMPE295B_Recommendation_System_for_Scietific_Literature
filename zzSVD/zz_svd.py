import numpy as np
import tensorflow as tf

# setting the random seed

np.random.seed(1000)


# Data Metrics

nb_users = 500	# users
nb_products = 200	# products
nb_factors = 50	# no of latent factors
max_rating = 5	# max possible rating for a paper
nb_rated_products = 5	# avg rated products per user
top_k_products = 10	# top products considered for evaluating the rating



# Create a random User-Item matrix
uim = np.zeros((nb_users, nb_products), dtype=np.float32)

for i in range(nb_users):
    nbp = np.random.randint(0, nb_products, size=nb_rated_products)
    for j in nbp:
        uim[i, j] = np.random.randint(1, max_rating+1)


#print(uim)


# Create a Tensorflow graph
graph = tf.Graph()

with graph.as_default():
	#User-item matrix
	user_item_matrix = tf.placeholder(tf.float32, shape=(nb_users, nb_products))

 	# SVD
	St, Ut, Vt = tf.svd(user_item_matrix)

    # Compute reduced matrices
	Sk = tf.diag(St)[0:nb_factors, 0:nb_factors]
	Uk = Ut[:, 0:nb_factors]
	Vk = Vt[0:nb_factors, :]

     # Compute Su and Si
	Su = tf.matmul(Uk, tf.sqrt(Sk))
	Si = tf.matmul(tf.sqrt(Sk), Vk)
    
    # Compute user ratings
	ratings_t = tf.matmul(Su, Si)
    
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
for i in range(10, 20):
    print('User {}: {}'.format(i, best_items[0][i]))