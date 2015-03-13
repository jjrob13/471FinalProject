from sklearn.naive_bayes import GaussianNB
import DataSet, Entry, threading, random

class NBThread(threading.Thread):
	def __init__(self, nb_num, training_population, sample_training_size):
		threading.Thread.__init__(self);
		self.NB_number = nb_num;
		self.training_population = training_population;
		self.sample_training_size = sample_training_size;
		self.classifier = [];
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