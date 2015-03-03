import Entry, numpy

class EntryCollection:
	def __get_median(self, feature_number):
		well_formed_features = [];
		for entry in self.entry_array:
			if(entry.feature_list[feature_number] != '?'):
				well_formed_features.append(entry.feature_list[feature_number]);

		#return 0 if there are no well formed features, ie. all features in the collection were '?', otherwise, return the median
		return 0 if len(well_formed_features) == 0 else numpy.median(well_formed_features);

	def __init__(self, entry_array):
		self.entry_array = entry_array;
		
		#vector of arithmetic medians for all features
		median_feature_vector = [];
		for i in range(len(entry_array[0].feature_list)):
			median_feature_vector.append(self.__get_median(i));

		#replace all '?' features with the arithmetic median of all entries
		for entry in self.entry_array:
			for i in range(len(entry.feature_list)):
				if entry.feature_list[i] == '?':
					#replace malformed input with arithmetic median of all other samples
					entry.feature_list[i] = median_feature_vector[i];


	def __str__(self):
		result_str = "";
		for entry in self.entry_array:
			result_str += "\n[ ";
			for feature in entry.feature_list:
				result_str += str(feature) + " ";
			result_str += "]";
		
		return result_str;

	#returns (X, Y)
	def get_X_Y_tuple(self):
		X = numpy.zeros(shape = (len(self.entry_array), len(self.entry_array[0].feature_list) - 1));

		"""Copy into X matrix"""
		for i in range(len(self.entry_array)):
			entry_features = [];
			#j is the number of features that have been copied for entry i
			for j in range(len(self.entry_array[i].feature_list) - 1):
				entry_features.append(self.entry_array[i].feature_list[j]);

			X[i] = numpy.array(entry_features);

		"""Copy into Y vector"""
		Y = numpy.zeros(shape = len(self.entry_array));
		for i in range(len(self.entry_array)):
			Y[i] = self.entry_array[i].feature_list[len(self.entry_array[i].feature_list) - 1];
		return (X, numpy.transpose(Y));


