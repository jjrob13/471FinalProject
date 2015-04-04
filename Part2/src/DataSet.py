import Entry, csv, random

class XY_Set:
	def __init__(self, x, y):
		self.x = x;
		self.y = y;	
class DataSet:
	def __init__(self, training_size_percentage = 0.5,  filename = 'crx.data.csv'):
		f = open(str(filename))

		csv_f = csv.reader(f);

		self.all_entries = [];
		for row in csv_f:
			self.all_entries.append(Entry.Entry(row));
		
		self.update_training_size(training_size_percentage);
	def update_training_size(self, training_size_percentage):
	
		random.shuffle(self.all_entries);
		training_size = int(training_size_percentage * len(self.all_entries)) ;
		
		training_data = self.all_entries[:training_size];
		test_data = self.all_entries[training_size:];
		
		self.testing_set = XY_Set(Entry.get_X_Y_tuple_from_entries(test_data)[0], Entry.get_X_Y_tuple_from_entries(test_data)[1]);			
		self.training_set = XY_Set(Entry.get_X_Y_tuple_from_entries(training_data)[0], Entry.get_X_Y_tuple_from_entries(training_data)[1]);	
