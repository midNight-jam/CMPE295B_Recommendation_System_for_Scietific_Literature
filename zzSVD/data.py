import numpy as np
import pandas as pd
from numpy import genfromtxt

# fname = "Data/small_users.dat"
fname = "Data/users.dat"
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
	 user_matrix = np.zeros(shape=(5552,16981)) # both have +1 dimension
	 user_id = 1
	 for line in open(fname):
	 	user_papers = map(int, line.split())

	 	for d in user_papers:
	 		user_matrix[user_id][d] = 1
	 	user_id += 1
	 return user_matrix

# X = read_user_data()
# X = read_user_data_ol()
# X = read_user_df()
# X = read_user_df()
X = read_and_create_user_Matrix()
print(X)
print(type(X))

# print(type(X[0][0]))
# print(type(X[0]))
# print(np.array_str(X[0]))
# print(["%.2f" % x for x in X[2]])