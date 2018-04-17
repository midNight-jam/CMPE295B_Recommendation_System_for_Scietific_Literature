import numpy as np
import pandas as pd
import numpy.random as random
from numpy import genfromtxt
import datetime
import math

fname = "Data/users.dat"
output_dir = "zzOutput/"
trimmed_users_count = 1865
trimmed_papers_count = 6000
threshold = 0.01
threshold_lib_size = 10
test_train_split = 0.25  # we do ceil to round
test_file = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)


# DO NOT USE THESE
# all these 3 functions below ARE USELESS, they will be removed
# they trim test and train file seperately which is not correct, but i am keeping the logic, incase if its used
def create_trimmed_test_train_data():
  create_trimmed_train_users_data()
  create_trimmed_test_users_data()


def create_trimmed_train_users_data():
  fname = "Data/cf-train-1-users.dat"
  name = "Data/trimmed-cf-train-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)
  create_trim_file(fname, name)


def create_trimmed_test_users_data():
  fname = "Data/cf-test-1-users.dat"
  name = "Data/trimmed-cf-test-1-users_{0}u_{1}p.dat".format(trimmed_users_count, trimmed_papers_count)
  create_trim_file(fname, name)


def create_trim_file(fname, name):
  user_id = 0
  trimmed_file = open(name, "w")
  for line in open(fname):
    docs = line.split()
    orig_count = docs.pop(0)  # removing original count
    trimmed_docs = []
    user_trimmed_docs_count = 0

    for d in docs:
      if (int(d) <= trimmed_papers_count):
        trimmed_docs.append(d)
        user_trimmed_docs_count += 1

    trimmed_docs.insert(0, user_trimmed_docs_count)  # adding new count
    trimmed_line = ' '.join(str(td) for td in trimmed_docs)
    trimmed_file.write(trimmed_line + "\n")

    user_id += 1
    if (user_id > trimmed_users_count):
      break
  trimmed_file.close()
