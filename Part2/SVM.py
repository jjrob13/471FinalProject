from sklearn import svm
import Entry, numpy, DataSet, random, threading


class SVMThread(threading.Thread):
	def __init__(self, svm_number, training_population, sample_training_size):
		threading.Thread.__init__(self);
		self.svm_number = svm_number;
		self.training_population = training_population;
		self.sample_training_size = sample_training_size;

	def run(self):
		random_sample = random.sample(set(self.training_population), self.sample_training_size);
		X_Y = Entry.get_X_Y_tuple_from_entries(random_sample);
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
		threadArray.append(newThread);

	svm_array = [];
	for thread in threadArray:
		thread.join();
		svm_array.append(thread.classifier);

	return svm_array;


