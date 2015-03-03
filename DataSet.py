import Entry, csv, random
class DataSet:
	def __init__(self, filename = 'crx.data.csv'):
		f = open(str(filename))

		csv_f = csv.reader(f);

		self.all_entries = [];
		for row in csv_f:
			self.all_entries.append(Entry.Entry(row));



	def get_training_and_test_set_tuple(self, training_set_size):
		if(training_set_size > len(self.all_entries)):
			raise ValueExcept('Size of training set must be less than or equal to population size');

		random.shuffle(self.all_entries);
		training_set = self.all_entries[0:training_set_size - 1];

		testing_set = self.all_entries[training_set_size:];

		return (training_set, testing_set);

