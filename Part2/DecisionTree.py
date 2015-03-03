from sklearn import tree
import Entry, numpy, DataSet, random, threading

class DecisionTreeThread(threading.Thread):
	def __init__(self, tree_id_number, training_population, sample_training_size):
		threading.Thread.__init__(self);
		self.tree_id_number = tree_id_number;
		self.training_population = training_population;
		self.sample_training_size = sample_training_size;

	def run(self):
		random_sample = random.sample(set(self.training_population), self.sample_training_size);
		X_Y = Entry.get_X_Y_tuple_from_entries(random_sample);
		X = X_Y[0];
		Y = X_Y[1];
		self.classifier = tree.DecisionTreeClassifier();
		self.classifier.fit(X, Y);

def create_decision_trees_concurrent(training_population, sample_training_size, number_of_trees):
	threadArray = [];
	#create and start all threads
	for i in range(number_of_trees):
		newThread = DecisionTreeThread(i, training_population, sample_training_size);
		newThread.start();
		threadArray.append(newThread);

	tree_array = [];
	for thread in threadArray:
		thread.join();
		tree_array.append(thread.classifier);

	return tree_array;
