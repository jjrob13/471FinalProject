from DataSet import DataSet
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
x = [];
y = [];
data = DataSet();
for i in range(1, 100):
	dt_bag = BaggingClassifier(DecisionTreeClassifier(), max_features = 0.5, n_estimators=i, max_samples=0.5);
	dt_bag.fit(data.training_set.x, data.training_set.y);
	x.append(i);
	y.append(100*dt_bag.score(data.testing_set.x, data.testing_set.y));

plt.plot(x, y, label='Decision Tree');
plt.legend(framealpha=0.5); 
plt.title("Number of Classifiers in Bagging vs Classification Correctness");
plt.suptitle("Training with 50% of Population");
plt.xlabel("# of Classifiers in Ensemble");
plt.ylabel("Percentage of Correct Classifications on Testing Data");
plt.show();	
