#-------------------------------------------------------------------------
# AUTHOR: Ian Chow
# FILENAME: knn.py
# SPECIFICATION: Find the error rate of the knn
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         db.append(row)

error_count = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):
    
    X = []
    Y = []
    
    for j in range(len(db)):
        if i != j:
            X.append([float(val) for val in db[j][:-1]]) 
            Y.append(1 if db[j][-1] == "spam" else 0)
    
    testSample = [float(val) for val in db[i][:-1]]  
    true_label = 1 if db[i][-1] == "spam" else 0

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction.
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        error_count += 1

#Print the error rate
error_rate = error_count / len(db)
print("Error Rate:", error_rate)
