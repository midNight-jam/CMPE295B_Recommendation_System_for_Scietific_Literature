import tensorflow as tf
import data
import config
import evaluations

def process(uim):
  print('shape of uim')
  print(uim.shape)
  print(uim.shape[0])
  print(uim.shape[1])
  print(type(uim.shape[0]))
  graph = tf.Graph()

  with graph.as_default():
    user_doc_matrix = tf.placeholder(tf.float32, shape=(config.numbers_users, config.numbers_papers))

    St, Ut, Vt = tf.svd(user_doc_matrix,  full_matrices=True)
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


    SLambda = tf.diag(St)[0:config.number_latent_factors, 0:config.number_latent_factors]
    ULambda = Ut[:, 0:config.number_latent_factors]
    VLambda = Vt[0:config.number_latent_factors, :]

    print('-' * 50)
    print('shape of SLambda')
    print(SLambda.shape)
    print('-' * 50)
    print('shape of ULambda')
    print(ULambda.shape)
    print('-' * 50)
    print('shape of VLambda')
    print(VLambda.shape)
    print('-' * 50)

    Suser = tf.matmul(ULambda, tf.sqrt(SLambda))
    Sdoc = tf.matmul(tf.sqrt(SLambda), VLambda)

    print('-' * 50)
    print('shape of Suser')
    print(Suser.shape)
    print('-' * 50)
    print('shape of Sdoc')
    print(Sdoc.shape)
    print('-' * 50)


    ratings_t = tf.matmul(Suser, Sdoc)
    print('-' * 50)
    print('shape of ratings_t')
    print(ratings_t.shape)
    print('-' * 50)
    best_ratings_t, best_items_t = tf.nn.top_k(ratings_t, config.top_k_products)


  session = tf.InteractiveSession(graph=graph)

  feed_dict = {
    user_doc_matrix: uim
  }

  best_items = session.run([best_items_t], feed_dict=feed_dict)

  text_file = open(config.svd_output_file, "w")
  for i in range(0, config.numbers_users):
    docs =  " ".join(str(d) for d in best_items[0][i])
    text_file.write('{0} {1}\n'.format(len(best_items[0][i]), docs))
  text_file.close()
  print(type(best_items))


  evaluations.process()


  return uim



########################################################################
# All below must be commented, debug only
########################################################################


# uim = data.read_and_create_user_Matrix()
# process(uim)


########################################################################
