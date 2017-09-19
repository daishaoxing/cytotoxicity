import numpy
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm, grid_search
from sklearn.externals import joblib
#####loaddata
alldataMat=joblib.load("alldata.pkl.z")
alldataClasses=joblib.load("alldatalabel.pkl.z")
#assigning predictor and target variables
x = numpy.array(alldataMat)
y = numpy.array(alldataClasses)
x_train, x_test, y_train, y_test=cross_validation.train_test_split(x,y,test_size=0.25)
############SVM
clf = svm.SVC(C=300,kernel='rbf',probability=True)
svm_model=clf.fit(x_train, y_train)
#joblib.dump(svm_model,'svm_model_300_rbf.pkl.z')
#svm_model=joblib.load("svm_model_300_rbf.pkl.z")
y_pred=[]
y_pred1= svm_model.predict_proba(x_test)
for i in y_pred1:
	y_pred.append(i[1])
svm_fpr, svm_tpr, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
svm_roc_auc=metrics.auc(svm_fpr, svm_tpr)
############RF
clf = RandomForestClassifier(max_depth=None,n_estimators=950,max_leaf_nodes=None,min_samples_split=2,min_samples_leaf=1)
RFmodel=clf.fit(x_train, y_train)
#joblib.dump(RFmodel,'RFmodel_950.pkl.z')
#RFmodel=joblib.load("RFmodel_950.pkl.z")
y_pred=[]
y_pred1= RFmodel.predict_proba(x_test)
for i in y_pred1:
	y_pred.append(i[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
rf_roc_auc=metrics.auc(rf_fpr, rf_tpr)

##### Plotting
plt.title('Receiver Operating Characteristic')

plt.plot(svm_fpr, svm_tpr, label=('SVM_AUC''= %0.4f'%svm_roc_auc),color='green', linewidth=2)
plt.plot(rf_fpr, rf_tpr, label=('RF_AUC''= %0.4f'%rf_roc_auc),color='blue', linewidth=2)

plt.legend(loc='lower right', prop={'size':8})
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
# Save ROC graphs
plt.savefig('svm_RF_roc.pdf',format='pdf')
