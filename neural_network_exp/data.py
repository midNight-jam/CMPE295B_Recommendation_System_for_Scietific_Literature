import numpy as np
import pandas as pd
from numpy import genfromtxt

# fname = "Data/small_users.dat"
# fname = "Data/users.dat"
fname = "Data/cf-train-1-users.dat"
# fname = "Data/fraction.dat"

trimmed_users_count = 500
trimmed_papers_count = 2500

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
	 	# # hv to find a better way to do this in python
	 	# skipMe = 0;
	 	for d in user_papers:
	 		# if skipMe == 0:
	 			# skipMe = 1
	 			# continue
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
	 		if(int(d) < trimmed_papers_count):
	 			trimmed_docs.append(d)
	 			user_trimmed_docs_count += 1
	 	
	 	trimmed_docs.insert(0, user_trimmed_docs_count)	# adding new count
	 	trimmed_line = ' '.join(str(td) for td in trimmed_docs)
	 	trimmed_file.write(trimmed_line+"\n")
	 	
	 	user_id += 1
	 	if(user_id > trimmed_users_count):
	 		break

	 trimmed_file.close()

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