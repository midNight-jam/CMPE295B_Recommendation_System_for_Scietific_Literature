import config
# trims the whole users.dat file into a smaller file
def trim_users_data():
  orig_file = 'data/ordered-users.dat'
  dest_file = 'tdata/t_ordered-users.dat'
  trimmed_file = open(dest_file, "w")
  for line in open(orig_file):
    docs = line.split()
    orig_count = int(docs.pop(0))

    trimmed_docs = []
    user_trimmed_docs_count = 0

    for d in docs:
      if (int(d) <= config.trimmed_papers_count):
        trimmed_docs.append(d)
        user_trimmed_docs_count += 1

    trimmed_line = ' '.join(str(td) for td in trimmed_docs)
    trimmed_file.write(trimmed_line + "\n")

  trimmed_file.close()


#############################################

trim_users_data()
#############################################

