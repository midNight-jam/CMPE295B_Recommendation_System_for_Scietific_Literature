import numpy as np
import pandas as pd
from numpy import genfromtxt

# fname = "Data/small_users.dat"
fname = "Data/users.dat"
# fname = "Data/cf-train-1-users.dat"
# fname = "Data/fraction.dat"

trimmed_users_count = 1500
trimmed_papers_count = 4500

def read_user_data():
	single=genfromtxt(fname,delimiter=' ', dtype=int)
	return single

def read_user_data_ol():
	result_array = np.empty([5551, 16980])
	arrays = [np.array(map(int, line.split())) for line in open(fname)]
	maxLen = 0
	tlen = 0
	for line in open(fname):
		tlen = len(line.split())
		if(tlen > maxLen):
			maxLen = tlen
		temp_arr = np.array(map(int, line.split()))
		result_array = np.append(result_array, [temp_arr], axis=0)

	print(maxLen)
	print('#'*10)
	return result_array
	# return np.vstack( arrays )
	# return np.concatenate( arrays, axis=0 )


def read_user_df():
	df = pd.read_csv(fname, sep="|")
	print (df)

#should return <class 'numpy.ndarray'> representation of the user matrix
def read_and_create_user_Matrix():
	 # user_matrix = np.zeros(shape=(5552,16981)) # both have +1 dimension
	 user_matrix = np.zeros((5551, 16980), dtype=np.float32)

	 user_id = 0
	 for line in open(fname):
	 	docs = line.split()
	 	docs.pop(0)
	 	user_papers = map(int, docs)
	 	for d in user_papers:
	 		user_matrix[user_id][d] = 1
	 	user_id += 1
	 return user_matrix

#should return <class 'numpy.ndarray'> representation of the user matrix
def read_and_create_trimmed_user_Matrix():
	 # user_matrix = np.zeros(shape=(5552,16981)) # both have +1 dimension
	 fname = "Data/trimmed_users.dat"
	 user_matrix = np.zeros((trimmed_users_count, trimmed_papers_count), dtype=np.int32)

	 user_id = 0
	 for line in open(fname):
	 	docs = line.split()
	 	docs.pop(0) # removing the paper count
	 	if(len(docs) > 0):
		 	user_papers = map(int, docs)
		 	for d in user_papers:
		 		user_matrix[user_id][d] = 1
		 	user_id += 1
	 return user_matrix


#should return <class 'numpy.ndarray'> representation of the user matrix
def read_and_create_paper_word_freq_Matrix():
	 no_papers = 16980
	 no_words = 8000
	 fname = "Data/mult.dat"
	 paper_vocab_matrix = np.zeros((no_papers, no_words), dtype=np.int32)
	 paper_id = 0
	 max_word_id = 0
	 for line in open(fname):
	 	docs = line.split()
	 	docs.pop(0)
	 	for d in docs:
	 		wf = d.split(':')
	 		if(len(wf)==2):
		 		word_id = int(wf[0])
		 		word_freq = int(wf[1]) 
		 		paper_vocab_matrix[paper_id][word_id] = word_freq
		 		if(word_id > max_word_id):
		 			max_word_id = word_id
	 	paper_id += 1
	 print('MAX Word ID : ' + str(max_word_id))
	 return paper_vocab_matrix


#should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_user_LibCount():
	 fname = "Data/users.dat"
	 user_lib = np.zeros(shape=(5552), dtype=np.int32) # both have +1 dimension
	 user_id = 1
	 for line in open(fname):
	 	docs = line.split()
	 	count = docs.pop(0)
	 	user_lib[user_id] = count
	 	user_id += 1
	 user_lib = np.delete(user_lib, 0)
	 return user_lib

#should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_paper_UserLibFreqCount():
	 fname = "Data/items.dat"
	 paper_lib = np.zeros(shape=(16981), dtype=np.int32) # both have +1 dimension
	 paper_id = 1
	 for line in open(fname):
	 	users = line.split()
	 	count = users.pop(0)
	 	paper_lib[paper_id] = count
	 	paper_id += 1
	 paper_lib = np.delete(paper_lib, 0)
	 return paper_lib

#should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_paper_wordFreqCount():
	 no_papers = 16980
	 no_words = 8000
	 fname = "Data/mult.dat"
	 paper_lib = np.zeros(shape=(no_papers + 1), dtype=np.int32)
	 paper_id = 1
	 for line in open(fname):
	 	users = line.split()
	 	count = users.pop(0)
	 	paper_lib[paper_id] = int(count)
	 	paper_id += 1
	 paper_lib = np.delete(paper_lib, 0)
	 print('SHAPE ::::: ', paper_lib.shape)
	 return paper_lib

#should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_word_paper_FreqCount():
	 fname = "Data/mult.dat"
	 max_word_freq = 0
	 words_dict = {}
	 for line in open(fname):
	 	docs = line.split()
	 	docs.pop(0)
	 	for d in docs:
	 		wf = d.split(':')
	 		if(len(wf)==2):
	 			word_id = int(wf[0])
		 		if(word_id in words_dict):
		 			words_dict[word_id] += 1
		 		else:
		 			words_dict[word_id] = 1
		 		if(words_dict[word_id] > max_word_freq):
		 			max_word_freq = words_dict[word_id]
	 print('MAX Word ID : ' + str(max_word_freq))
	 return words_dict


def create_trimmed_users_data():
	 fname = "Data/users.dat"
	 user_id = 1
	 trimmed_file = open("Data/trimmed_users.dat","w")
	 
	 for line in open(fname):
	 	docs = line.split()
	 	orig_count = docs.pop(0)	# removing original count
	 	trimmed_docs = []
	 	user_trimmed_docs_count = 0

	 	for d in docs:
	 		if(int(d) <= trimmed_papers_count):
	 			trimmed_docs.append(d)
	 			user_trimmed_docs_count += 1
	 	
	 	trimmed_docs.insert(0, user_trimmed_docs_count)	# adding new count
	 	trimmed_line = ' '.join(str(td) for td in trimmed_docs)
	 	trimmed_file.write(trimmed_line+"\n")
	 	
	 	user_id += 1
	 	if(user_id > trimmed_users_count):
	 		break

	 trimmed_file.close()

def read_generated_csv():
	 fname = "out_mar_30_1500u_4500p.csv"
	 rec_data = genfromtxt(fname, delimiter=',')
	 # user_id = 1
	 # low = 0.3
	 # high = 1.1
	 # non_zero_rec = np.ma.masked_outside(rec_data,low,high)
	 # non_zero_rec = np.asarray(non_zero_rec)
	 threshold = 0.98
	 rec_file = open("non_zero_rec_gen_mar_30_1500u_4500p.dat","w")
	 for r in rec_data:
	 	doc_id = 1
	 	rec_docs = []
	 	for p in r:
	 		if(p > threshold):
	 			rec_docs.append(doc_id)
	 		doc_id += 1
	 	rec_docs.insert(0, len(rec_docs))	# adding new count
	 	line_str = ' '.join(str(d) for d in rec_docs)
	 	rec_file.write(line_str+"\n")
	 rec_file.close()

# X = read_user_data()
# X = read_user_data_ol()
# X = read_user_df()
# X = read_user_df()
# X = read_and_create_user_Matrix()
# print(X)
# print(X.shape)
# print(type(X))
# print(X[0])

# # print(type(X[0][0]))
# # print(type(X[0]))
# # print(np.array_str(X[0]))
# print(["%d" % x for x in X[2]])

# create_trimmed_users_data()
# print(read_and_create_trimmed_user_Matrix().shape)

# read_generated_csv()

print(read_and_create_paper_wordFreqCount()[0])