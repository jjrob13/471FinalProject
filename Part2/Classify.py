import SVM, NaiveBayes, DataSet, Entry, DecisionTree, threading, random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def predict_sample_with_bagging(x, classifier_array):
	total_sum = 0;
	for classifier in classifier_array:
		total_sum = total_sum + classifier.predict(x)[0];

	return round((total_sum * 1.0)/len(classifier_array));

def test_mixed_bagging_on_set(X, Y, classifier_array):
	i = 0;
	correct = 0;
	#i is the number of samples that have been tested
	while i != len(X):
		if(predict_sample_with_bagging(X[i], classifier_array) == Y[i]):
			correct = correct + 1;
		i = i + 1;

	return (correct * 100.0)/i;


class Learner(threading.Thread):
	def __init__(self, learning_func_to_call, training_population, sampling_size, num_learners):
		threading.Thread.__init__(self);

		self.learning_func_to_call = learning_func_to_call;
		self.num_learners = num_learners;
		self.training_population = training_population;
		self.sampling_size = sampling_size;
		self.classifiers = [];

	def run(self):
		self.classifiers = self.learning_func_to_call(self.training_population, self.sampling_size, self.num_learners);

class TesterWorker(threading.Thread):
	def __init__(self, X, Y, classifiers):
		threading.Thread.__init__(self);
		self.classifiers = classifiers;
		self.X = X;
		self.Y = Y;
		self.result = [];
	def run(self):
		self.result = test_mixed_bagging_on_set(self.X, self.Y, self.classifiers);

class Tester(threading.Thread):
	def __init__(self, X, Y, classifiers):
		threading.Thread.__init__(self);
		self.classifiers = classifiers;
		self.X = X;
		self.Y = Y;
		self.y_vals = [];
	def run(self):
		workerThreads = [];
		for i in range(1,len(self.classifiers)+1):
			sample_of_classifiers = random.sample(set(self.classifiers), i);
			newWorkerThread = TesterWorker(self.X, self.Y, sample_of_classifiers);
			workerThreads.append(newWorkerThread);
			newWorkerThread.start();

		for t in workerThreads:
			t.join();
		self.y_vals = [];
		for t in workerThreads:
			self.y_vals.append(t.result);


def test_and_plot(num_svm = 3, num_nb = 10, num_dt = 10, training_set_size = 400, sampling_size = 100, plot_aggregate = True):
	data = DataSet.DataSet();
	train_test = data.get_training_and_test_set_tuple(training_set_size);
	training_set = train_test[0];
	testing_set = train_test[1];

	X_Y = Entry.get_X_Y_tuple_from_entries(training_set);
	X = X_Y[0];
	Y = X_Y[1];

	learnerThreads = [Learner(SVM.create_svms_concurrent, training_set, sampling_size, num_svm),
	 Learner(NaiveBayes.create_NBs_concurrent, training_set, sampling_size, num_nb),
	 Learner(DecisionTree.create_decision_trees_concurrent, training_set, sampling_size, num_dt)];


	for t in learnerThreads:
		t.start();

	for t in learnerThreads:
		t.join();

	testThreads = [];
	for t in learnerThreads:
		testThreads.append(Tester(X, Y, t.classifiers));
	for t in testThreads:
		t.start();
	for t in testThreads:
		t.join();

	allClassifiers = [];
	for t in learnerThreads:
		allClassifiers.extend(t.classifiers);

	allClassifierTester = Tester(X, Y, allClassifiers);
	allClassifierTester.start();
	allClassifierTester.join();

	#plot results for each individual learner
	if num_svm != 0:
		plt.plot([0] + testThreads[0].y_vals, label='SVM');
	
	if num_nb != 0:
		plt.plot([0] + testThreads[1].y_vals, label='Naive Bayes');
	
	if num_dt != 0:
		plt.plot([0] + testThreads[2].y_vals, label='Decision Tree');


	if plot_aggregate:
		plt.plot([0] + allClassifierTester.y_vals, label='Aggregation of all methods');

	#set axis range
	plt.xlim(1, plt.xlim()[1]);
	plt.ylim(75, plt.ylim()[1]);
	plt.legend(framealpha=0.5);

	plt.ylabel("Correct classification (%)");
	plt.xlabel("Number of classifiers in Bagging");
	plt.suptitle("Correctness vs number of learners in Bag");
	plt.title("Sample Size: " + str(sampling_size) + "\nTraining Population: " + str(len(training_set)));

	plt.show();

test_and_plot(0, 0, 10);