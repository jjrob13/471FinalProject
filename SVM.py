from sklearn import svm
import csv, EntryCollection, Entry, numpy

f = open('crx.data.csv')

csv_f = csv.reader(f);

test_entry_array = [];
validation_entry_array = [];
i = 0;
for row in csv_f:
	if i % 2 == 0:
		test_entry_array.append(Entry.Entry(row));
	else:
		validation_entry_array.append(Entry.Entry(row));
	i = i + 1;

test_collection = EntryCollection.EntryCollection(test_entry_array);
validation_collection = EntryCollection.EntryCollection(validation_entry_array);


clf = svm.SVC();

X_Y_test = test_collection.get_X_Y_vector_tuple();

X = X_Y_test[0];
Y = numpy.transpose(X_Y_test[1]);

clf.fit(X, Y);


X_Y_validation = validation_collection.get_X_Y_vector_tuple();


X_val = X_Y_validation[0];
Y_val = numpy.transpose(X_Y_validation[1]);
correct = 0;

i = 0;
while i != len(X_val):
	if clf.predict(X_val[i])[0] == Y_val[i]:
		correct = correct + 1;

	i = i + 1;

print (correct * 1.0)/i * 100;