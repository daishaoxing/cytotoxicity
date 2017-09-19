from sklearn.externals import joblib
import pybel
import cPickle
import random

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print "the word: %s is not in my Vocabulary!" % word
	return returnVec

def get_bits(f1):
	mols = list(pybel.readfile("smi",f1));
	titles = [x.title for x in mols]
	bits = [x.calcfp().bits for x in mols]
	return bits,titles

def numtostringlist(inlist):
	list1=[]
	for i in inlist:
		a=str(i)
		list1.append(a)
	return list1

activebits,activetitles=get_bits('cytotoxicity_active_final.smi')
#####
f1=open('cytotoxicity_inactive_final.smi','r')
f2=open('cytotoxicity_inactive_final_select.smi','w')
aa=f1.readlines()
random_inactive=random.sample(aa, len(activebits))
f2.write(''.join(random_inactive))
f1.close()
f2.close()
####
inactivebits,inactivetitles=get_bits('cytotoxicity_inactive_final_select.smi')
print len(activebits)
print len(inactivebits)

fingerprint=range(1,1025)
alldata=[]
alldatalabel=[]
for i in activebits:
	returnVec=setOfWords2Vec(fingerprint,i)
	list1=numtostringlist(returnVec)
	alldata.append(returnVec)
	alldatalabel.append(1)

###
for i in inactivebits:
	returnVec=setOfWords2Vec(fingerprint,i)
	list1=numtostringlist(returnVec)
	alldata.append(returnVec)
	alldatalabel.append(0)
data_and_label=[]
for i in range(len(alldata)):
	a=[alldata[i],alldatalabel[i]]
	data_and_label.append(a)
alldata=[]
alldatalabel=[]
random.shuffle(data_and_label)
for i in range(len(data_and_label)):
	alldata.append(data_and_label[i][0])
	alldatalabel.append(data_and_label[i][1])
#cPickle.dump(alldata,open("alldata.pkl","wb"))#data = cPickle.load(open("alldata.pkl","rb"))
#cPickle.dump(alldatalabel,open("alldatalabel.pkl","wb"))
joblib.dump(alldata,'alldata.pkl.z') #data = joblib.load('alldata.pkl.z')
joblib.dump(alldatalabel,'alldatalabel.pkl.z')
