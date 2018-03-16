import numpy as np
import pandas as pd
from numpy import genfromtxt

# fname = "Data/small_users.dat"
# fname = "Data/users.dat"
fname = "Data/cf-train-1-users.dat"
# fname = "Data/fraction.dat"
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
	 user_matrix = np.zeros((552, 16981), dtype=np.float32)

	 user_id = 1
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

#should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_user_LibCount():
	 user_lib = np.zeros(shape=(5552,1)) # both have +1 dimension
	 # user_matrix = np.zeros((552, 16981), dtype=np.float32)

	 user_id = 1
	 for line in open(fname):
	 	docs = line.split()
	 	first = docs.pop(0)
 		user_lib[user_id][0] = first
	 	user_id += 1
	 return user_lib

# X = read_user_data()
# X = read_user_data_ol()
# X = read_user_df()
# X = read_user_df()
# X = read_and_create_user_LibCount()
# print(X)
# print(X.shape)
# print(type(X))
# print(X[0][0])

# print(type(X[0][0]))
# print(type(X[0]))
# print(np.array_str(X[0]))
# print(["%.2f" % x for x in X[2]])