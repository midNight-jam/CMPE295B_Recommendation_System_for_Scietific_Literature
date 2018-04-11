import numpy as np
import pandas as pd
from numpy import genfromtxt
import datetime

# fname = "Data/small_users.dat"
fname = "Data/users.dat"
# fname = "Data/cf-train-1-users.dat"
# fname = "Data/fraction.dat"
output_dir = "zzOutput/"
trimmed_users_count = 2000
trimmed_papers_count = 6000
threshold = 0.02
test_file = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)

def read_user_data():
	single=genfromtxt(fname,delimiter=' ', dtype=int)
	return single

def read_user_df():
	df = pd.read_csv(fname, sep="|")
	print (df)

#should return <class 'numpy.ndarray'> representation of the user matrix
def read_and_create_user_Matrix():
	 user_matrix = np.zeros((5550, 16980), dtype=np.float32)

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
	 name = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)
	 user_matrix = np.zeros((trimmed_users_count, trimmed_papers_count), dtype=np.int32)

	 user_id = 0
	 for line in open(name):
	 	docs = line.split()
	 	docs.pop(0) # removing the paper count
	 	if(len(docs) > 0):
		 	user_papers = map(int, docs)
		 	for d in user_papers:
		 		user_matrix[user_id][d] = 1
		 	user_id += 1
	 return user_matrix


def create_trimmed_train_users_data():
	 fname = "Data/cf-train-1-users.dat"
	 name = "Data/trimmed-cf-train-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)
	 create_trim_file(fname, name)

def create_trimmed_test_users_data():
	 fname = "Data/cf-test-1-users.dat"
	 name = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)
	 create_trim_file(fname, name)

def create_trim_file(fname, name):
	 user_id = 1
	 trimmed_file = open(name,"w")
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
	 paper_lib = np.zeros(shape=(no_papers), dtype=np.int32)
	 paper_id = 0
	 max_count = 0
	 for line in open(fname):
	 	users = line.split()
	 	count = int(users.pop(0))
	 	paper_lib[paper_id] = count
	 	if(count > max_count):
	 		max_count = count
	 	paper_id += 1
	 print('SHAPE ::::: {} \n max count ::: {}'.format(paper_lib.shape, max_count))
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
	 print('MAX Word Frequency : ' + str(max_word_freq))
	 return words_dict


def read_generated_csv(right_now):
	 fname = "out__"+right_now+".csv"
	 rec_data = genfromtxt(fname, delimiter=',')
	 gen_pred_file = output_dir+'gen__pred__docs'+right_now+'.dat'
	 rec_file = open(gen_pred_file,"w")
	 for r in rec_data:
	 	doc_id = 0
	 	rec_docs = []
	 	for p in r:
	 		if(p > threshold):
	 			rec_docs.append(doc_id)
	 		doc_id += 1
	 	rec_docs.insert(0, len(rec_docs))	# adding new count
	 	line_str = ' '.join(str(d) for d in rec_docs)
	 	rec_file.write(line_str+"\n")
	 rec_file.close()

def read_generated_csv_dictionary(right_now):
	 rec_data = genfromtxt('out__'+right_now+'.csv', delimiter=',')
	 # right_now = str(datetime.datetime.now().isoformat())
	 gen_pred_file = output_dir+'gen__pred__sorted__'+right_now+'.dat'
	 rec_file = open(gen_pred_file,"w")

	 for r in rec_data:
	 	pred = {}
	 	doc_id = 0
	 	for p in r:
	 		if(p >= threshold):
	 			pred[doc_id] = p
	 		doc_id += 1

	 	keys_sorted_by_value_pred = sorted(pred, key=pred.get, reverse=True)
	 	line_str = str(len(pred))	# adding new count

	 	for k in keys_sorted_by_value_pred:
	 		line_str+= ' '+ str(k) + ':' + str(pred[k])

	 	rec_file.write(line_str+"\n")

	 rec_file.close()



def read_generated_user_test_pred_dictionary(right_now):
	 pred_file = "out__"+right_now+".csv"
	 pred_data = genfromtxt(pred_file, delimiter=',')
	 pred_user_dict = {}
	 pred_user_dict_paper_info = {}
	 user_id = 1;

	 for pred in pred_data:
	 	pred_user_dict[user_id] = []
	 	pred_user_dict_paper_info[user_id] = {}
	 	doc_id = 0
	 	for p in pred:
	 		if(p >= threshold):
	 			pred_user_dict[user_id].append( doc_id)
	 			pred_user_dict_paper_info[user_id][doc_id]= p
	 		doc_id += 1
	 	user_id += 1

	 user_id = 1;
	 test_user_dict = {}

	 for line in open(test_file):
	 	docs = line.split()
	 	docs.pop(0)
	 	user_papers = map(int, docs)
	 	test_user_dict[user_id] = []
	 	
	 	for d in user_papers:
	 		test_user_dict[user_id].append(d)
	 	user_id += 1

	 return pred_user_dict, test_user_dict, pred_user_dict_paper_info


def preicsion(right_now):
	 pred, test, pred_tuples = read_generated_user_test_pred_dictionary(right_now)
	 if(len(pred) != len(test) or len(pred) != len(pred_tuples)):
	 	print("Lengths of predicted & test users dictionary doesnt match by row Count")
	 	return
	 
	 user_count = len(pred)
	 final_precision = 0.0
	 max_precision = 0.0
	 included_users = 0

	 for i  in range(user_count):
	 	i += 1
	 	users_likes_count = len(pred[i])
	 	
	 	if(users_likes_count < 1):
	 		# print("Skipping for user {} as no predictions found".format(i))
	 		continue

	 	included_users+=1
	 	pred_matches_count = 0
	 	test_users_doc = test[i]
	 	test_users_set = set(test_users_doc)
	 	pred_users_doc = pred[i]
	 	
	 	for pd in pred_users_doc:
	 		if( pd in test_users_set):
	 			pred_matches_count += 1

	 	user_precision = pred_matches_count / users_likes_count
	 	if(user_precision > max_precision):
	 		max_precision = user_precision

	 	print(' u '+ str(i)+'- p '+str(user_precision))

	 	final_precision += user_precision

	 print('-'*50)
	 print('final Precision / included users')
	 print('{0} / {1}'.format(final_precision, included_users))
	 final_precision = final_precision / included_users

	 print('Total users - {0}'.format(user_count))
	 print('Users included - {0}'.format(included_users))
	 print('Max Precision : ' + str(max_precision))
	 print('Final Precision : ' + str(final_precision))

	 precision_output_name = output_dir+'precision__'+right_now+'.dat'
	 precision_file = open(precision_output_name,"w")
	 precision_file.write('final Precision / included users\n')
	 precision_file.write('{0} / {1}\n'.format(final_precision, included_users))
	 precision_file.write('Total users - {0}\n'.format(user_count))
	 precision_file.write('Users included - {0}\n'.format(included_users))
	 precision_file.write('Max Precision : {0}\n'.format(max_precision))
	 precision_file.write('Final Precision : {0}\n'.format(final_precision))
	 precision_file.close()

def preicsion_M(right_now):
	 #Precision@M = # items the user likes in the list / M
	 pred, test, pred_info = read_generated_user_test_pred_dictionary(right_now)
	 if(len(pred) != len(test) or len(pred) != len(pred_info)):
	 	print("Lengths of predicted & test users dictionary doesnt match by row Count")
	 	return
	 
	 user_count = len(pred)
	 final_precision_M = 0.0
	 max_precision_M = 0.0
	 included_users = 0
	 M = 10

	 for i  in range(user_count):
	 	i += 1 # as user ids begin @ 1
	 	users_likes_count = len(pred[i])
	 	
	 	if(users_likes_count < M):
	 		# print("Skipping for user {} as no predictions found".format(i))
	 		continue

	 	included_users+=1
	 	pred_matches_count = 0
	 	test_users_doc = test[i]
	 	test_users_set = set(test_users_doc)
	 	pred_users_doc = pred[i]
	 	count = 0
	 	keys_sorted_by_value_pred = sorted(pred, key=pred.get, reverse=True)

	 	for pd in keys_sorted_by_value_pred:
	 		
	 		if( pd in test_users_set):
	 			pred_matches_count += 1
	 		count+=1
	 		if(count >= M):
	 			break

	 	user_precision_M = pred_matches_count / M
	 	if(user_precision_M > max_precision_M):
	 		max_precision_M = user_precision_M

	 	print(' u {0} - p {1}'.format(i, user_precision_M))

	 	final_precision_M += user_precision_M

	 print('-'*50)
	 print('Preicision @ M - {}'.format(M))
	 print('-'*50)
	 print('final Precision / included users')
	 print('{0} / {1}'.format(final_precision_M, included_users))
	 final_precision_M = final_precision_M / included_users

	 print('Total users - {0}'.format(user_count))
	 print('Users included - {0}'.format(included_users))
	 print('Max Precision @ {0}- : {1}'.format(M, max_precision_M))
	 print('Final Precision @ {0} : {1}'.format(M,final_precision_M))

	 precision_M_output_name = output_dir+'precision__@_M_'+right_now+'.dat'
	 precision_M_file = open(precision_M_output_name,"w")
	 precision_M_file.write('Precision @ M\n M = {0}\n'.format(M))
	 precision_M_file.write('final Precision / included users\n')
	 precision_M_file.write('{0} / {1}\n'.format(final_precision_M, included_users))
	 precision_M_file.write('Total users - {0}\n'.format(user_count))
	 precision_M_file.write('Users included - {0}\n'.format(included_users))
	 precision_M_file.write('Max Precision @ {0} : {1}\n'.format(M, max_precision_M))
	 precision_M_file.write('Final Precision @ {0} : {1}\n'.format(M,final_precision_M))
	 precision_M_file.close()


def recall(right_now):
	 pred, test, pred_info = read_generated_user_test_pred_dictionary(right_now)
	 if(len(pred) != len(test) or len(pred) != len(pred_info)):
	 	print("Lengths of predicted & test users dictionary doesnt match by row Count")
	 	return
	 
	 user_count = len(pred)
	 final_recall = 0.0
	 max_recall = 0.0
	 included_users = 0

	 for i  in range(user_count):
	 	i += 1
	 	users_relevant_docs_count = len(test[i])
	 	
	 	# if there are no predictions or no relevant docs in test(divide by 0 else)
	 	if(len(pred[i]) < 1 or users_relevant_docs_count < 1):
	 		# print("Skipping for user {} as no predictions found".format(i))
	 		continue

	 	included_users+=1
	 	pred_matches_count = 0
	 	test_users_doc = test[i]
	 	test_users_set = set(test_users_doc)
	 	pred_users_doc = pred[i]
	 	
	 	for pd in pred_users_doc:
	 		if( pd in test_users_set):
	 			pred_matches_count += 1

	 	user_recall = pred_matches_count / users_relevant_docs_count
	 	if(user_recall > max_recall):
	 		max_recall = user_recall

	 	print(' u {0} - p {1}'.format(i, user_recall))

	 	final_recall += user_recall

	 print('-'*50)
	 print('final Precision / included users')
	 print('{0} / {1}'.format(final_recall, included_users))
	 final_recall = final_recall / included_users

	 print('Total users - {0}'.format(user_count))
	 print('Users included - {0}'.format(included_users))
	 print('Max Recall : ' + str(max_recall))
	 print('Final Recall : ' + str(final_recall))
	 
	 recall_output_name = output_dir+'recall__'+right_now+'.dat'
	 recall_file = open(recall_output_name,"w")
	 recall_file.write('final Recall / included users\n')
	 recall_file.write('{0} / {1}\n'.format(final_recall, included_users))
	 recall_file.write('Total users - {0}\n'.format(user_count))
	 recall_file.write('Users included - {0}\n'.format(included_users))
	 recall_file.write('Max Recall : {0}\n'.format(max_recall))
	 recall_file.write('Final Recall :{0}\n'.format(final_recall))
	 recall_file.close()


def get_cruve_readings(readings_file):
	 # readings_file = "loss_plot__2018-04-09T02:48:29.095769.dat"
	 readings = []
	 for line in open(readings_file):
	 	readings.append(float(line))
	 return readings

########################################################################
# All below must be commented, debug only
########################################################################

# X = read_user_data()
# X = read_user_data_ol()
# X = read_user_df()
# X = read_user_df()
# X = read_and_create_user_Matrix()

# create_trimmed_users_data()
# X = read_and_create_trimmed_user_Matrix() 
# print(X.shape)

# read_generated_csv()
# #
# print(read_generated_csv())
# print(read_generated_csv_dictionary())


# p, t, pin = read_generated_user_test_pred_dictionary()
# print('pred len '+str(len(p)))
# print('test len '+str(len(t)))
# print(p[21])
# print(t[21])
# print(pin[21])

# create_trimmed_train_users_data()
# create_trimmed_test_users_data()

read_and_create_paper_wordFreqCount()

#============================================
# run after the out.csv is generated
#============================================
# right_now = "2018-04-09T19:13:04.874678"
# read_generated_csv_dictionary(right_now)
# read_generated_csv(right_now)
# preicsion(right_now)
# preicsion_M(right_now)
# recall(right_now)
#============================================

########################################################################