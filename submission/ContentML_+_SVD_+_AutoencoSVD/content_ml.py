import data
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

print(config.content_ml_rec_output_file)

mat = data.read_and_create_term_frequency()
tf_transformer = TfidfTransformer().fit(mat)
tf_idf_mat = tf_transformer.transform(mat)
tf_arr = tf_idf_mat.toarray()
print('tf_idf_mat......')
print(tf_idf_mat.shape)
cosine_similarities = linear_kernel(tf_idf_mat, tf_idf_mat)
print('Cosine Similarites calculated')
print(cosine_similarities.shape)
results = {}

for id in range(0, 16980):
  similar_indices = cosine_similarities[id].argsort()[:-50:-1]
  similar_items = [(cosine_similarities[id][i], i) for i in similar_indices]

	# First item is the item itself, so remove it.
	# Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
  results[id] = similar_items[1:]

print('Done calculating the similar ones')
print('----'*8)
print(results[0])
print('----'*8)

content_ml_rec_details_output_file = open(config.content_ml_rec_details_output_file, "w")
content_ml_rec_output_file = open(config.content_ml_rec_output_file, "w")
print("-------"*5)

for i in range(0, 16980):
  item_id = i
  recs = results[item_id][:config.top_k_products]
  user_rec = []
  user_rec_papers = []
  for rec in recs:
    user_rec.append((rec[1],rec[0]))
    user_rec_papers.append(rec[1])
  print("u : {0}   rec_papers {1}".format(item_id, str(user_rec_papers)))
  content_ml_rec_details_output_file.write("u : {0}   rec_papers {1}\n".format(item_id, str(user_rec)))
  docs = " ".join(str(d) for d in user_rec_papers)
  content_ml_rec_output_file.write("{0} {1}\n".format(len(user_rec_papers), docs))

content_ml_rec_details_output_file.close()
content_ml_rec_output_file.close()

######################################################

######################################################