from sklearn.naive_bayes import GaussianNB
import DataSet, Entry, threading, random

class NBThread(threading.Thread):
	def __init__(self, nb_num, training_population, sample_training_size):
		threading.Thread.__init__(self);
		self.NB_number = nb_num;
		self.training_population = training_population;
		self.sample_training_size = sample_training_size;

	def run(self):
		random_sample = random.sample(set(self.training_population), self.sample_training_size);
		X_Y = Entry.get_X_Y_tuple_from_entries(random_sample);
		X = X_Y[0];
		Y = X_Y[1];
		self.classifier = GaussianNB();
		self.classifier.fit(X, Y);

def create_NBs_concurrent(training_population, sample_training_size, number_of_NBs):
	threadArray = [];
	#create and start all threads
	for i in range(number_of_NBs):
		newThread = NBThread(i, training_population, sample_training_size);
		newThread.start();
		threadArray.append(newThread);

	nb_array = [];
	for thread in threadArray:
		thread.join();
		nb_array.append(thread.classifier);

	return nb_array;

def predict_sample_with_NB_bagging(x, NB_array):
	total_sum = 0;
	for NB in NB_array:
		total_sum = total_sum + NB.predict(x)[0];

	return round((total_sum * 1.0)/len(NB_array));

def test_NB_bagging_on_set(X, Y, nb_array):
	i = 0;
	correct = 0;
	#i is the number of samples that have been tested
	while i != len(X):
		if(predict_sample_with_NB_bagging(X[i], nb_array) == Y[i]):
			correct = correct + 1;
		i = i + 1;

	return (correct * 100.0)/i;

TRAINING_SIZE = 600;
NUM_NBS = 50;
SAMPLE_SIZE = 50;

data = DataSet.DataSet();

train_test = data.get_training_and_test_set_tuple(TRAINING_SIZE);
training_set = train_test[0];
testing_set = train_test[1];


bayes_classifiers = create_NBs_concurrent(training_set, SAMPLE_SIZE, NUM_NBS);

X_Y = Entry.get_X_Y_tuple_from_entries(testing_set);
X,Y = X_Y[0], X_Y[1];

print test_NB_bagging_on_set(X, Y, bayes_classifiers);