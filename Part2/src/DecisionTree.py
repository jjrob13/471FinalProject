from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import Entry, numpy, DataSet, random, threading
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = DataSet.DataSet();
num_estimators = 100;
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
		dt_bag = BaggingClassifier(DecisionTreeClassifier(), max_features = i, n_estimators=num_estimators, max_samples=j);
		dt_bag.fit(data.training_set.x, data.training_set.y);
		z.append(100*dt_bag.score(data.testing_set.x, data.testing_set.y));

fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
ax.plot_wireframe(x, y, z);
ax.set_xlabel('Features Given to Each Classifier (%)');
ax.set_ylabel('Samples Given to Each Classifier (%)');
ax.set_zlabel('Classification Score (%)');
ax.set_title('Decision Tree Bagging With 100 Classifiers, 50% Training Size.\n # Features, # Samples vs Score');
plt.show();	
