import glob
import matplotlib.pyplot as plt

def getUserItemMatrix(line):
	line = line.strip()
	rowList = [int(x) for x in line.rstrip('\n').split(" ")]
	return(rowList[0])

userItem=[]
for name in glob.glob('rec-list.dat'):
	for line in open(name,'r+'):
		temp = line.strip()
		stArray = temp.split(":")
		userItem.append(int(stArray[0]))

totalDocsPerUser=[]
for line in open('../ordered-cf-test-1-users.dat', 'r+'):
	totalDocsPerUser.append(getUserItemMatrix(line))


precision = [x/10 for x in userItem]
recall=[]
loop=0
for x in userItem:
	if(totalDocsPerUser[loop]!=0):
		recall.append(x/(totalDocsPerUser[loop]))
	else:
		recall.append(0)
	
	loop = loop+1

plt.plot(precision)
plt.ylabel('precision')
plt.xlabel('users in decreasing order of density ->')
plt.show()

plt.plot(recall)
plt.ylabel('recall')
plt.xlabel('users in decreasing order of density ->')
plt.show()

