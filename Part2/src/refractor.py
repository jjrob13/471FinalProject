import DataSet
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

data = DataSet.DataSet();
axes = [];
for i in range(1, 10):
	data.update_training_size(i * .1);
	svm = SVC(kernel = 'linear');
	svm.fit(data.training_set.x, data.training_set.y);
	svm_score = svm.score(data.testing_set.x, data.testing_set.y);

	nb = GaussianNB();
	nb.fit(data.training_set.x, data.training_set.y);
	nb_score = nb.score(data.testing_set.x, data.testing_set.y);


	dt = DecisionTreeClassifier();
	dt.fit(data.training_set.x, data.training_set.y);
	dt_score = dt.score(data.testing_set.x, data.testing_set.y);

	fig = plt.figure(i);
	ax = fig.add_subplot(111);
	x = range(3);
	rects = [];
	rects.append(ax.bar(0, svm_score, color='r', label="SVM"));
	rects.append(ax.bar(1, nb_score, color='b', label="NB"));
	rects.append(ax.bar(2, dt_score, color='g', label="DT"));
	labels = ("", "SVM", "",  "NB", "", "DT");
	ax.set_xticklabels(labels);
	ax.set_title("Training With {0}% of Data".format(str(i * 10)))
	axes.append(ax);
plt.show();
