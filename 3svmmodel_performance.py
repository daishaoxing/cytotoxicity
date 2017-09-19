from sklearn import metrics
from sklearn import svm, grid_search
from sklearn import cross_validation
import numpy
import cPickle
import threading
from sklearn.externals import joblib
import sys
###########
def run_svm_thread(x_train,x_test,y_train,y_test,k,C,cv):
		clf = svm.SVC(C=C,kernel=k,probability=True)
		svm_model=clf.fit(x_train, y_train)
		y_pred= svm_model.predict(x_test)
		accuracy=metrics.accuracy_score(y_test, y_pred)
		precision=metrics.precision_score(y_test, y_pred)
		recall=metrics.recall_score(y_test, y_pred)
		f1score=metrics.f1_score(y_test,y_pred)
		y_pred1=[]
		y_pred2= svm_model.predict_proba(x_test)
		for i in y_pred2:
			y_pred1.append(i[1])
		fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred1, pos_label=1)
		roc_auc=metrics.auc(fpr, tpr)
		strb=str(C)+'\t'+str(cv)+'\t'+str(accuracy)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1score)+'\t'+str(roc_auc)+'\n'
		f1.write(strb)
		print strb
############main
f1=open('svm_performance_final.txt','a')
alldataMat=joblib.load("alldata.pkl.z")
alldataClasses=joblib.load("alldatalabel.pkl.z")
#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)
k='rbf'
C=float (sys.argv[1])
for cv in range(10):
	x_train, x_test, y_train, y_test=cross_validation.train_test_split(x,y,test_size=0.25)
	myargs = (x_train, x_test, y_train, y_test,k,C,cv,)
	print k
	print C
	print cv
	threading.Thread(target=run_svm_thread, args=myargs).start()
