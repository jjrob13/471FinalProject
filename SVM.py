from sklearn import svm
import csv, EntryCollection, Entry, numpy

f = open('crx.data.csv')

csv_f = csv.reader(f);

entry_array = [];

for row in csv_f:
	entry_array.append(Entry.Entry(row));

collection = EntryCollection.EntryCollection(entry_array);



clf = svm.SVC();

X_Y = collection.get_X_Y_vector_tuple();

X = X_Y[0];
Y = numpy.transpose(X_Y[1]);

clf.fit(X, Y);
