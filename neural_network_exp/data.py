import numpy as np
import pandas as pd
import numpy.random as random
from numpy import genfromtxt
import datetime
import math
from sklearn.feature_extraction.text import TfidfTransformer
from operator import itemgetter
from numpy import genfromtxt
import config


# should return <class 'numpy.ndarray'> representation of the user matrix
def read_and_create_user_Matrix():
  user_matrix = np.zeros((config.numbers_users, config.numbers_papers), dtype=np.float32)
  user_id = 0
  for line in open( config.train_file):
    docs = line.split()
    docs.pop(0)
    user_papers = map(int, docs)
    for d in user_papers:
      user_matrix[user_id][d] = 1
    user_id += 1
  return user_matrix


# should return <class 'numpy.ndarray'> representation of the user matrix
# creates a sparse matrix of trimmed users.dat file
def read_and_create_trimmed_user_Matrix(name):
  user_matrix = np.zeros((config.trimmed_users_count, config.trimmed_papers_count), dtype=np.int32)

  user_id = 0
  for line in open(name):
    docs = line.split()
    docs.pop(0)  # removing the paper count
    if (len(docs) > 0):
      user_papers = map(int, docs)
      for d in user_papers:
        user_matrix[user_id][d] = 1
      user_id += 1
  return user_matrix


# should return <class 'numpy.ndarray'> representation of the user matrix
# creates a sparse matrix of trimmed users.dat file, but including only users that have a library size of >= config.rating_threshold
def read_and_create_trimmed_user_Matrix_Threshold():
  name = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(config.trimmed_users_count, config.trimmed_papers_count)
  user_matrix = np.zeros((config.trimmed_users_count, config.trimmed_papers_count), dtype=np.int32)

  user_id = 0
  for line in open(name):
    docs = line.split()
    docs.pop(0)  # removing the paper count
    if (len(docs) > config.lib_size_threshold):
      user_papers = map(int, docs)
      for d in user_papers:
        user_matrix[user_id][d] = 1
    user_id += 1
  return user_matrix


# trims the whole users.dat file into a smaller file
def trim_users_data(fname, name):
  trimmed_file = open(name, "w")
  for line in open(fname):
    docs = line.split()
    orig_count = int(docs.pop(0))

    trimmed_docs = []
    user_trimmed_docs_count = 0

    for d in docs:
      if (int(d) <= config.trimmed_papers_count):
        trimmed_docs.append(d)
        user_trimmed_docs_count += 1

    # if (user_trimmed_docs_count >= config.lib_size_threshold):
    trimmed_docs.insert(0, user_trimmed_docs_count)  # adding new count
    trimmed_line = ' '.join(str(td) for td in trimmed_docs)
    trimmed_file.write(trimmed_line + "\n")

  trimmed_file.close()


# splits the input users.dat file in to test and train files, using the split % "test_train_split" defined abover
def split_users_data(dname):
  data_name = dname.split('.');
  train_name = data_name[0] + '_train' + '_' + str(config.test_train_split) + '_.' + data_name[1];
  test_name = data_name[0] + '_test' + '_' + str(config.test_train_split) + '_.' + data_name[1];

  train_file = open(train_name, "w")
  test_file = open(test_name, "w")

  for line in open(dname):
    docs = line.split()
    orig_count = int(docs.pop(0))

    random.shuffle(docs)
    total = len(docs)
    sep_out = math.ceil(config.test_train_split * total)
    test_docs = []

    for i in range(sep_out):
      test_docs.append(docs.pop())

    train_docs_count = len(docs)
    test_docs_count = len(test_docs)

    docs.insert(0, train_docs_count)  # adding new count to test data
    test_docs.insert(0, test_docs_count)  # adding new count to train data

    train_line = ' '.join(str(td) for td in docs)
    test_line = ' '.join(str(td) for td in test_docs)

    train_file.write(train_line + "\n")
    test_file.write(test_line + "\n")

  train_file.close()
  test_file.close()


# should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_term_frequency():
  fname = "Data/mult.dat"
  term_frequency = np.zeros(shape=(16980, 8000), dtype=np.int32)  # both have +1 dimension

  doc_id = 0

  for line in open(fname):
    wf_tuple = line.split()
    wf_tuple.pop(0)
    for wf in wf_tuple:
      word_freq = wf.split(':')

      if (len(word_freq) == 2):
        id = int(word_freq[0])
        freq = int(word_freq[1])
        term_frequency[doc_id][id] = freq
    doc_id += 1

  return term_frequency


def tf_idf_papers_vs_words_freq():
  tf = read_and_create_term_frequency();
  tf_transformer = TfidfTransformer().fit(tf)
  tf_idf_mat = tf_transformer.transform(tf)
  tf = tf_idf_mat.toarray()
  print(tf_idf_mat.shape)
  return tf

# should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_document_frequency():
  fname = "Data/mult.dat"
  max_word_freq = 0
  words_dict = {}
  for line in open(fname):
    docs = line.split()
    docs.pop(0)
    for d in docs:
      wf = d.split(':')
      if (len(wf) == 2):
        word_id = int(wf[0])
        if (word_id in words_dict):
          words_dict[word_id] += 1
        else:
          words_dict[word_id] = 1
        if (words_dict[word_id] > max_word_freq):
          max_word_freq = words_dict[word_id]
  print('MAX Word Frequency : ' + str(max_word_freq))
  return words_dict


def read_generated_csv(right_now):
  fname = "out__" + right_now + ".csv"
  rec_data = genfromtxt(fname, delimiter=',')
  gen_pred_file = config.output_dir + 'gen__pred__docs' + right_now + '.dat'
  rec_file = open(gen_pred_file, "w")
  for r in rec_data:
    doc_id = 0
    rec_docs = []
    for p in r:
      if (p > config.rating_threshold):
        rec_docs.append(doc_id)
      doc_id += 1
    rec_docs.insert(0, len(rec_docs))  # adding new count
    line_str = ' '.join(str(d) for d in rec_docs)
    rec_file.write(line_str + "\n")
  rec_file.close()


def read_generated_csv_dictionary(right_now):
  rec_data = genfromtxt('out__' + right_now + '.csv', delimiter=',')
  # right_now = str(datetime.datetime.now().isoformat())
  gen_pred_file = config.output_dir + 'gen__pred__sorted__' + right_now + '.dat'
  rec_file = open(gen_pred_file, "w")

  for r in rec_data:
    pred = {}
    doc_id = 0
    for p in r:
      if (p >= config.rating_threshold):
        pred[doc_id] = p
      doc_id += 1

    keys_sorted_by_value_pred = sorted(pred, key=pred.get, reverse=True)
    line_str = str(len(pred))  # adding new count

    for k in keys_sorted_by_value_pred:
      line_str += ' ' + str(k) + ':' + str(pred[k])

    rec_file.write(line_str + "\n")

  rec_file.close()


def read_generated_user_test_pred_dictionary(right_now):
  pred_file = "out__" + right_now + ".csv"
  pred_data = genfromtxt(pred_file, delimiter=',')
  pred_user_dict = {}
  pred_user_dict_paper_info = {}
  user_id = 0;
  test_file = "Data/trim/users_5551_papers_8500.dat"
  for pred in pred_data:
    pred_user_dict[user_id] = []
    pred_user_dict_paper_info[user_id] = {}
    doc_id = 0
    for p in pred:
      if (p >= config.rating_threshold):
        pred_user_dict[user_id].append(doc_id)
        pred_user_dict_paper_info[user_id][doc_id] = p
      doc_id += 1
    user_id += 1

  user_id = 0;
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



def get_cruve_readings(readings_file):
  # readings_file = "loss_plot__2018-04-09T02:48:29.095769.dat"
  readings = []
  for line in open(readings_file):
    docs = line.split()
    val = float(docs[4])
    readings.append(val)
  return readings



########################################################################
# All below must be commented, debug only
########################################################################

# train_file_name = "Data/trim/users_5551_papers_6000_libsize_15_train_0.25_.dat"
# X = read_and_create_trimmed_user_Matrix(train_file_name, config.trimmed_users_count, config.trimmed_papers_count)
# str = X[0].tolist()
# print(X.shape)
# print(str)

# X = read_and_create_user_Matrix()
# print(X.shape)
# print(str(X[0]))
# # # str = np.array2string(X[21], precision=2, separator=',',suppress_small=True)


# name = "Data/trim/users_5551_papers_{0}.dat".format(config.trimmed_papers_count)
# trim_users_data(source_data_file, name);

# dname = "Data/trim/users_5551_papers_{0}_libsize_{1}.dat".format(config.trimmed_papers_count, config.lib_size_threshold)
# split_users_data(dname)

# read_generated_csv()
# #
# print(read_generated_csv())
# print(read_generated_csv_dictionary())

# trim_users_data('Data/users.dat', 'Data/trim_users_5551_6000_papers_.dat')
# split_users_data('Data/trim_users_5551_6000_papers_.dat')
# trim_users_data(source_data_file, )

# p, t, pin = read_generated_user_test_pred_dictionary()
# print('pred len '+str(len(p)))
# print('test len '+str(len(t)))
# print(p[21])
# print(t[21])
# print(pin[21])

# create_trimmed_train_users_data()
# create_trimmed_test_users_data()

# read_and_create_paper_wordFreqCount()

# tf = tf_idf_papers_vs_words_freq()
# print(type(tf))
# # print(tf.shape)
# # s = str(tf[0])
# # print(s)
# print(' bat ' * 5)
# batches = np.array_split(tf, 3)
# print(str(batches[0][0]))
# df = read_and_create_document_frequency()
# print(df)
# print(df[0])
# print(" zzzzzz "*5)

# pfile = 'zzOutput/recall__2018-04-17T15:26:34.483896.dat'
# R = get_cruve_readings(pfile)
# print(R[0])
# ============================================
# run after the out.csv is generated
# ============================================
# right_now = "2018-04-18T17:06:03.667176"
# pred, test, pred_tuples = read_generated_user_test_pred_dictionary(right_now)
# read_generated_csv_dictionary(right_now)
# read_generated_csv(right_now)
# preicsion(pred, test, pred_tuples, right_now)
# preicsion_M(pred, test, pred_tuples, right_now)
# recall(pred, test, pred_tuples, right_now)
# ============================================

########################################################################
