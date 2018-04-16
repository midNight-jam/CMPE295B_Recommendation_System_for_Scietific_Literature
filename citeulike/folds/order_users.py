import glob
from operator import itemgetter
import pickle
import pandas as pd


def lineCount(name):
	count=0
	for line in open(name, 'r+'):
		count = count+1
	return count

def getUserItemMatrix(line, list):
	line.strip()
	rowList = [int(x) for x in line.rstrip('\n').split(" ")]
	list.append(rowList)
	return list

def writeToFile(fname, matrix):
	f = open(fname, 'w+')
	for row in matrix:
		for inner_row in row:
			f.write(str(inner_row) + " ")
		f.write('\n')


for name in glob.glob('cf-test-*-users.dat'):
	print(name)
	userItem =[]
	for line in open(name, 'r+'):
		userItem = getUserItemMatrix(line, userItem)
	userItem.sort(key=itemgetter(0), reverse=True)
	my_df = pd.DataFrame(userItem)
	writeToFile('ordered-' + name, userItem)

	print(userItem[:5])
	print(userItem[5500:])




