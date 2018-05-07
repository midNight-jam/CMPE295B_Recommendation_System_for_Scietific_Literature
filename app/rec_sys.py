from beautifultable import BeautifulTable

doc_info_file = 'data/raw-data.csv'
content_ml_output_file = 'data/content_ml_app_out.dat'

most_rec = 3

def read_doc_info(docId):
  i = 0
  docs = []
  for line in open(doc_info_file):
    docs = line.split(",")
    if (i == docId):
      break
    i += 1
  return docs[3]

def read_content_ml_output(userId):
  i = 0
  docs = []
  for line in open(content_ml_output_file):
    docs = line.split()
    if(i == userId):
      break
    i += 1
  top_rec = docs[:most_rec]
  top_rec = list(map(int, top_rec))
  return top_rec

def get_all_docs(recs):
  table = BeautifulTable()
  table.column_headers = ["paperId", "title"]
  for r in recs:
    table.append_row([r, read_doc_info(r)])
  print(table)
  return table


def decoratedRow(str):
  print('-'*50)
  print(str)
  print('-'*50)

def start():
  decoratedRow('Welcome To Recommendation System')
  validInput = False
  userId  = 0
  while validInput != True:
    userId = input('Enter a userId between {0} - {1} \n'.format(1, 5551))
    userId = int(userId)
    if(userId < 1 or userId > 5551):
      print('Wrong userId')
      validInput = False
    else:
      validInput = True

  print('User ID : {0}'.format(userId))
  userId -=1 # as all IDS are 0 based
  content_ml_rec = read_content_ml_output(userId)
  # print(content_ml_rec)
  res = get_all_docs(content_ml_rec)
  # print(res)


start()