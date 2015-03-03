from sklearn.naive_bayes import GaussianNB
import DataSet, Entry


TRAINING_SIZE = 400;
data = DataSet.DataSet();

train_test = data.get_training_and_test_set_tuple(TRAINING_SIZE);
training_set = train_test[0];
testing_set = train_test[1];

X_Y = Entry.get_X_Y_tuple_from_entries(training_set);
X = X_Y[0];
Y = X_Y[1];

bayes_classifier = GaussianNB();

bayes_classifier.fit(X, Y);