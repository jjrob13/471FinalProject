from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
import Entry, numpy, DataSet, random, threading
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bag_plot_3d(classifier, fig, num_estimators = 100):
	data = DataSet.DataSet();
	x = [];
	y = [];
	z = [];
	total_features = 15.0;
	# Percentage of Features
	for i in (x/total_features for x in range(1, int(total_features))): 
		# Percentage of training_data to present to each model
		for j in (x*.01 for x in range(1, 101) if x % 5 is 0):	
		
			x.append(100*i);
			y.append(100*j);
			bag = BaggingClassifier(classifier(), max_features = i, n_estimators=num_estimators, max_samples=j);
			bag.fit(data.training_set.x, data.training_set.y);
			z.append(100*bag.score(data.testing_set.x, data.testing_set.y));

	ax = fig.add_subplot(111, projection='3d');
	ax.plot_wireframe(x, y, z);
	return ax;

ax1 = bag_plot_3d(DecisionTreeClassifier, plt.figure(1));
ax1.set_xlabel('Features Given to Each Classifier (%)');
ax1.set_ylabel('Samples Given to Each Classifier (%)');
ax1.set_zlabel('Classification Score (%)');
ax1.set_title('Decision Tree Bagging With 100 Classifiers, 50% Training Size.\n # Features, # Samples vs Score');


ax2 = bag_plot_3d(GaussianNB, plt.figure(2));
ax2.set_xlabel('Features Given to Each Classifier (%)');
ax2.set_ylabel('Samples Given to Each Classifier (%)');
ax2.set_zlabel('Classification Score (%)');

ax2.set_title('Naive Bayes Bagging With 100 Classifiers, 50% Training Size.\n # Features, # Samples vs Score');
plt.show();	
