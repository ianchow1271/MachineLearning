#-------------------------------------------------------------------------
# AUTHOR: Ian Chow
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test a decision tree classifier on different datasets
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 Hours
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY SUCH AS numpy OR pandas.
# Only use standard Python libraries like dictionaries, lists, and arrays.

from sklearn import tree
import csv

def transform_features(row):
    # Mapping categorical features to numerical values
    feature_mapping = {
        "Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
        "Myope": 1, "Hypermetrope": 2,
        "Yes": 1, "No": 2,
        "Reduced": 1, "Normal": 2
    }
    return [feature_mapping[val] for val in row[:-1]]

def transform_label(label):
    # Mapping class labels to numerical values
    return 1 if label == "Yes" else 2

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

test_data = []
# Reading test data
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        test_data.append(row)

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading training data
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            dbTraining.append(row)
            X.append(transform_features(row))
            Y.append(transform_label(row[-1]))
    
    total_accuracy = 0
    
    # Running the training and testing 10 times
    for _ in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        correct_predictions = 0
        
        # Testing the classifier
        for data in test_data:
            X_test = transform_features(data)
            y_true = transform_label(data[-1])
            y_pred = clf.predict([X_test])[0]
            
            if y_pred == y_true:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_data)
        total_accuracy += accuracy
    
    average_accuracy = total_accuracy / 10
    print(f'Final accuracy when training on {ds}: {average_accuracy:.3f}') 
