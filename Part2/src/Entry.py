import numpy, MyStringLib, EntryCollectionProcessor

class Entry:
	"""Static class variable.  Lists that are used to convert from char to int.  It essentially simulates enums"""
	feature_hash = { "F0" : ['a', 'b'], "F3" : ['u', 'y', 'l', 't'], "F4" : ['g', 'p', 'gg'],
	"F5" : ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
	"F6" : ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], "F8" : ['t', 'f'], "F9" : ['t', 'f'],
	"F11" : ['t', 'f'], "F12" : ['g', 'p', 's'], "F15" : ['-', '+']
	};

	@staticmethod

	#helper method to convert feature into its numerical equivalent.
	def process_feature_char(feature_number, str_to_convert):
		#if the string is numerical, return it.  If it is already a number it can be used to learn.
		if(MyStringLib.is_number(str_to_convert)):
			return float(str_to_convert);

		#the character '?' is an appropriate character and represents a missing feature
		if(str_to_convert == '?'):
			return str_to_convert;

		#refers to the static feature lists defined above.  They essentially act as enums.
		list_name = 'F' + str(feature_number);
		if(list_name in Entry.feature_hash and str_to_convert in Entry.feature_hash[list_name]):
			return Entry.feature_hash[list_name].index(str_to_convert);

		#malformed input, treat as missing feature
		return '?';

	#constructor that takes a feature vector, filled with strings and converts it to a numerical featured vector
	#which can be used to learn models.
	def __init__(self, str_feature_list):
		#processed feature list
		self.feature_list = [];
		for i in range(len(str_feature_list)):
			self.feature_list.append(self.process_feature_char(i, str_feature_list[i]));


	#to string method
	def __str__(self):
		result_str = "[ ";
		for feature in self.feature_list:
			result_str += str(feature) + " ";

		result_str += "]";
		return result_str;

def get_X_Y_tuple_from_entries(entry_array):
	collection = EntryCollectionProcessor.EntryCollection(entry_array);
	return collection.get_X_Y_tuple();