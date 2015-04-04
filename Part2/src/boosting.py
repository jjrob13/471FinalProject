from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from DataSet import DataSet
from matplotlib import pyplot as plt
data = DataSet();

x = [];
ada_y = [];
gbc_y = [];
N_ESTIMATORS = 1000;
for i in range(1, N_ESTIMATORS + 1):
	data.shuffle();
	x.append(i);
	ada = AdaBoostClassifier(n_estimators=i);
	ada.fit(data.training_set.x, data.training_set.y);
	ada_y.append(ada.score(data.testing_set.x, data.testing_set.y));
	gbc = GradientBoostingClassifier(n_estimators=i);
	gbc.fit(data.training_set.x, data.training_set.y);
	gbc_y.append(gbc.score(data.testing_set.x, data.testing_set.y));

plt.plot(x, ada_y, label="AdaBoost");
plt.plot(x, gbc_y, label="GradientBoostingClassifier");
plt.title("Number of Estimators vs Score");
plt.suptitle("Training with 50% of Population");
plt.legend(framealpha=0.25);
plt.xlabel("Estimators");
plt.ylabel("Score");
plt.show();	
