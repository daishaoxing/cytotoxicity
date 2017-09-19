import time
start = time.strftime('%b %d %Y %H:%M:%S',time.localtime(time.time()))
import numpy
from sklearn.externals import joblib
import pybel
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def mol_calcfp(f1):
		mols =  list(pybel.readfile("smi",f1))
		bits = [x.calcfp() for x in mols]
		titles = [x.title for x in mols]
		return bits,titles

def mol_calcfpbit(f1):
		mols =  list(pybel.readfile("smi",f1))
		bits = [x.calcfp().bits for x in mols]
		titles = [x.title for x in mols]
		return bits,titles
		
def cal_psore(f1,f2):
		testbit,testtitles=mol_calcfp(f1)
		cabit,catitles=mol_calcfp(f2)
		fpscore={}
		for i in xrange(len(testtitles)):
				atc1={}
				for j in xrange(len(catitles)):
						tascore=testbit[i] | cabit[j]
						atc1[catitles[j]]=tascore
				atclist= sorted(atc1.iteritems(), key=lambda d:d[1], reverse = True)
				atc=atclist[0][1]
				matc=atclist[0][0]
				fpscore[testtitles[i]]=(matc,atc)
		return fpscore

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print "the word: %s is not in my Vocabulary!" % word
	return returnVec

###main
FPscore_dict={}
FPscoret_dict={}
SVM_dict={}
RF_dict={}
SVMa_dict={}
RFa_dict={}
f1='all_approved_drug.smi' ####query drugs
f2='cytotoxicity_active_final.smi'
f3=open('cytotoxicity_prediction.txt','w')
header="query\tMatch_ID\tMax_TC\tSVM_Prob\tSVM_pre\tRF_Prob\tRF_Pre\n"
f3.write(header)

fpscore=cal_psore(f1,f2)
bits,titles=mol_calcfpbit(f1)
fingerprint=range(1,1025)
querydata=[]
for i in bits:
	returnVec=setOfWords2Vec(fingerprint,i)
	querydata.append(returnVec)
x = numpy.array(querydata)

for i in range(len(titles)):
	FPscore_dict[titles[i]]=fpscore[titles[i]][1]
	FPscoret_dict[titles[i]]=fpscore[titles[i]][0]

###SVM
svm_model=joblib.load("cytotoxicity_svm_model_300_rbf.pkl.z")
y_pred= svm_model.predict(x)
y_pred1= svm_model.predict_proba(x)
for i in range(len(y_pred1)):
	y_predp=y_pred1[i][1]
	SVM_dict[titles[i]]=y_predp
	SVMa_dict[titles[i]]=y_pred[i]

###RF
RFmodel=joblib.load("cytotoxicity_RFmodel_950.pkl.z")
y_pred= RFmodel.predict(x)
y_pred1= RFmodel.predict_proba(x)
for i in range(len(y_pred1)):
	y_predp=y_pred1[i][1]
	RF_dict[titles[i]]=y_predp
	RFa_dict[titles[i]]=y_pred[i]

for i in FPscore_dict.keys():
	stra='\t'.join([str(i),str(FPscoret_dict[i]),str(FPscore_dict[i]),str(SVM_dict[i]),str(SVMa_dict[i]),str(RF_dict[i]),str(RFa_dict[i])])+'\n'
	f3.write(stra)
end = time.strftime('%b %d %Y %H:%M:%S',time.localtime(time.time()))
f3.write("Start Time: " + start + "\n")
f3.write("End Time: " + end + "\n")
