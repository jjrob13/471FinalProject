import DataSet
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

data = DataSet.DataSet();
x = [];
svm_y = [];
nb_y = [];
dt_y = [];
for i in range(1, 10):
	x.append(10*i);
	data.update_training_size(i * .1);
	svm = SVC(kernel = 'linear');
	svm.fit(data.training_set.x, data.training_set.y);
	svm_y.append(100* svm.score(data.testing_set.x, data.testing_set.y));

	nb = GaussianNB();
	nb.fit(data.training_set.x, data.training_set.y);
	nb_y.append(100*nb.score(data.testing_set.x, data.testing_set.y));


	dt = DecisionTreeClassifier();
	dt.fit(data.training_set.x, data.training_set.y);
	dt_y.append(100*dt.score(data.testing_set.x, data.testing_set.y));

plt.plot(x, svm_y, label="SVM")
plt.plot(x, nb_y, label="Naive Bayes");
plt.plot(x, dt_y, label = "Decision Tree");
plt.legend(framealpha=0.5);
plt.title("Population Training Percentage vs Correct Classification Score");
plt.xlabel("Percentage of Population Used for Training");
plt.ylabel("Classification Correctness (%)");
plt.suptitle('Single Classifier Comparison');
plt.show();