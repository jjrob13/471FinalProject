from sklearn.naive_bayes import GaussianNB
import DataSet, EntryCollection, Entry


TRAINING_SIZE = 400;
data = DataSet.DataSet();

train_test = data.get_training_set_and_test_set_tuple(TRAINING_SIZE);
training_set = train_test[0];
testing_set = train_test[1];

bayes_classifier = GaussianNB();

bayes_classifier.fit(training_set[0], training_set[1]);

correct = 0;
for i in range(len(testing_set[0])):
	if bayes_classifier.predict(testing_set[0][i]) == testing_set[1][i]:
		correct = correct + 1;

print (correct * 100.0)/len(testing_set[0]);


correct = 0;
for i in range(len(training_set[0])):
	if bayes_classifier.predict(training_set[0][i]) == training_set[1][i]:
		correct = correct + 1;

print (correct * 100.0)/len(training_set[0]);