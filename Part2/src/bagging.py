from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from DataSet import DataSet
data = DataSet();
num_classifiers = 300;
x = range(1, num_classifiers + 1);
svm_y = [];
dt_y = [];
nb_y = [];
# i is number of models to learn this iteration
for i in x:
	# Only learning 100 svms
	if i <= 100:
		svm = BaggingClassifier(SVC(kernel='linear'), n_estimators=i);
		svm.fit(data.training_set.x, data.training_set.y);
		svm_y.append(svm.score(data.testing_set.x, data.testing_set.y));
	nb = BaggingClassifier(GaussianNB(), max_samples=0.4, n_estimators=i);
	nb.fit(data.training_set.x, data.training_set.y);
	nb_y.append(nb.score(data.testing_set.x, data.testing_set.y));
	dt = BaggingClassifier(DecisionTreeClassifier(), n_estimators=i);
	dt.fit(data.training_set.x, data.training_set.y);
	dt_y.append(dt.score(data.testing_set.x, data.testing_set.y));


plt.plot(x, svm_y, label='SVM');
plt.plot(x, dt_y, label='Decision Tree');
plt.plot(x, nb_y, label="Naive Bayes");
plt.legend(framealpha=0.25);
plt.title('Number of Classifiers vs Score');
plt.suptitle('Training with 50% of Population');
plt.show();
