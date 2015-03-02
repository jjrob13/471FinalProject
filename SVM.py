from sklearn import svm
import EntryCollection, Entry, numpy, DataSet, random

def create_svms(training_population, sample_training_size, number_of_svms):
	svm_array = [];
	for i in range(number_of_svms):
		random_sample = random.sample(set(training_population), sample_training_size);
		training_collection = EntryCollection.EntryCollection(random_sample);
		X_Y = training_collection.get_X_Y_vector_tuple();
		X = X_Y[0];
		Y = X_Y[1];
		clf = svm.SVC(kernel = 'linear');
		clf.fit(X, Y);
		svm_array.append(clf);

	return svm_array;

def predict_sample_with_svm_bagging(x, svm_array):
	total_sum = 0;
	for svm in svm_array:
		total_sum = total_sum + svm.predict(x)[0];

	return round((total_sum * 1.0)/len(svm_array));

def test_svm_bagging_on_set(X, Y, svm_array):
	i = 0;
	correct = 0;
	#i is the number of samples that have been tested
	while i != len(X):
		if(predict_sample_with_svm_bagging(X[i], svm_array) == Y[i]):
			correct = correct + 1;
		i = i + 1;

	return (correct * 100.0)/i;


TRAINING_SIZE = 400;
NUM_SVMS = 1;
SAMPLE_SIZE = 100;

data = DataSet.DataSet();
train_test = data.get_training_set_and_test_set_tuple(TRAINING_SIZE);
training_set = train_test[0];
testing_set = train_test[1];

svms = create_svms(training_set, SAMPLE_SIZE, NUM_SVMS);	

testing_collection = EntryCollection.EntryCollection(testing_set);
X_Y = testing_collection.get_X_Y_vector_tuple();
X = X_Y[0];
Y = X_Y[1];

print test_svm_bagging_on_set(X, Y, svms);