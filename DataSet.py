import Entry, EntryCollection, csv, random
class DataSet:
	def __init__(self, filename = 'crx.data.csv'):
		f = open(str(filename))

		csv_f = csv.reader(f);

		self.all_entries = [];
		for row in csv_f:
			self.all_entries.append(Entry.Entry(row));



	def get_training_set_and_test_set_tuple(self, training_set_size):
		if(training_set_size > len(self.all_entries)):
			raise ValueExcept('Size of training set must be less than or equal to population size');

		random.shuffle(self.all_entries);
		training_set = self.all_entries[0:training_set_size - 1];

		testing_set = self.all_entries[training_set_size:];

		training_collection = EntryCollection.EntryCollection(training_set);
		testing_collection = EntryCollection.EntryCollection(testing_set);

		return (training_collection.get_X_Y_vector_tuple(), testing_collection.get_X_Y_vector_tuple());

