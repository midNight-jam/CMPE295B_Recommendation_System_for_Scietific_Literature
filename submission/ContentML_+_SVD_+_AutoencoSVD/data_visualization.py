import numpy as np
import pandas as pd
import numpy.random as random
from numpy import genfromtxt
import datetime
import math

# should return <class 'numpy.ndarray'> representation of the user matrix
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
      if (len(wf) == 2):
        word_id = int(wf[0])
        word_freq = int(wf[1])
        paper_vocab_matrix[paper_id][word_id] = word_freq
        if (word_id > max_word_id):
          max_word_id = word_id
    paper_id += 1
  print('MAX Word ID : ' + str(max_word_id))
  return paper_vocab_matrix


# should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_user_LibCount():
  fname = "Data/users.dat"
  user_lib = np.zeros(shape=(5552), dtype=np.int32)  # both have +1 dimension
  user_id = 0
  for line in open(fname):
    docs = line.split()
    count = docs.pop(0)
    user_lib[user_id] = count
    user_id += 1
  user_lib = np.delete(user_lib, 0)
  return user_lib


# should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_paper_UserLibFreqCount():
  fname = "Data/items.dat"
  paper_lib = np.zeros(shape=(16981), dtype=np.int32)  # both have +1 dimension
  paper_id = 0
  for line in open(fname):
    users = line.split()
    count = users.pop(0)
    paper_lib[paper_id] = count
    paper_id += 1
  paper_lib = np.delete(paper_lib, 0)
  return paper_lib


# should return <class 'numpy.ndarray'> representation of the user library count
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
    if (count > max_count):
      max_count = count
    paper_id += 1
  print('SHAPE ::::: {} \n max count ::: {}'.format(paper_lib.shape, max_count))
  return paper_lib


# should return <class 'numpy.ndarray'> representation of the user library count
def read_and_create_word_paper_FreqCount():
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
