from sklearn import svm
import Entry, numpy, DataSet, random, threading

def create_svms(training_population, sample_training_size, number_of_svms):
	svm_array = [];
	for i in range(number_of_svms):
		random_sample = random.sample(set(training_population), sample_training_size);
		X_Y = Entry.get_X_Y_tuple_from_entries(random_sample);
		X = X_Y[0];
		Y = X_Y[1];
		classifier = svm.SVC(kernel = 'linear');
		classifier.fit(X, Y);
		svm_array.append(classifier);

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

class SVMThread(threading.Thread):
	def __init__(self, svm_number, training_population, sample_training_size):
		threading.Thread.__init__(self);
		self.svm_number = svm_number;

	def run(self):
		random_sample = random.sample(set(training_population), sample_training_size);
		X_Y = training_collection.get_X_Y_tuple_from_entries(random_sample);
		X = X_Y[0];
		Y = X_Y[1];
		self.classifier = svm.SVC(kernel = 'linear');
		self.classifier.fit(X, Y);

def create_svms_concurrent(training_population, sample_training_size, number_of_svms):
	threadArray = [];
	#create and start all threads
	for i in range(number_of_svms):
		newThread = SVMThread(i, training_population, sample_training_size);
		newThread.start();
		threadArray.append();

	svm_array = [];
	for thread in threadArray:
		thread.join();
		svm_array.append(thread.classifier);

	return svm_array;


TRAINING_SIZE = 400;
NUM_SVMS = 1;
SAMPLE_SIZE = 100;

data = DataSet.DataSet();
train_test = data.get_training_and_test_set_tuple(TRAINING_SIZE);
training_set = train_test[0];
testing_set = train_test[1];

svms = create_svms(training_set, SAMPLE_SIZE, NUM_SVMS);	

X_Y = Entry.get_X_Y_tuple_from_entries(training_set);
X = X_Y[0];
Y = X_Y[1];

print test_svm_bagging_on_set(X, Y, svms);