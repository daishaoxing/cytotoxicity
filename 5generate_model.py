import numpy
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.externals import joblib
#####loaddata
alldataMat=joblib.load("alldata.pkl.z")
alldataClasses=joblib.load("alldatalabel.pkl.z")
#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)

############SVM
clf = svm.SVC(C=300,kernel='rbf',probability=True)
svm_model=clf.fit(x, y)
joblib.dump(svm_model,'cytotoxicity_svm_model_300_rbf.pkl.z')
############RF
clf = RandomForestClassifier(max_depth=None,n_estimators=950,max_leaf_nodes=None,min_samples_split=2,min_samples_leaf=1)
RFmodel=clf.fit(x, y)
joblib.dump(RFmodel,'cytotoxicity_RFmodel_950.pkl.z')
