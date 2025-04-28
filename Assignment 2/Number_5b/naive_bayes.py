#-------------------------------------------------------------------------
# AUTHOR: Ian Chow
# FILENAME: naive_bayes.py
# SPECIFICATION: Naïve Bayes classifier for weather data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Define mappings for categorical data
outlook_map = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_map = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_map = {"High": 1, "Normal": 2}
wind_map = {"Weak": 1, "Strong": 2}
play_tennis_map = {"Yes": 1, "No": 2}

#Reading the training data in a csv file
training_data = []
labels = []

with open("./weather_training.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        training_data.append([
            outlook_map[row[1]], temperature_map[row[2]], humidity_map[row[3]], wind_map[row[4]]
        ])
        labels.append(play_tennis_map[row[5]])

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(training_data, labels)

#Reading the test data in a csv file
test_data = []
test_instances = []

with open("./weather_test.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        test_instances.append(row[0])  
        test_data.append([
            outlook_map[row[1]], temperature_map[row[2]], humidity_map[row[3]], wind_map[row[4]]
        ])

# Printing the header of the solution
print("Day, Outlook, Temperature, Humidity, Wind, PlayTennis")

# Making predictions
predictions = clf.predict(test_data)
probabilities = clf.predict_proba(test_data)

# Reopen the test file to retrieve the original feature values
with open("./weather_test.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  
    test_rows = list(reader)  

# Output the classification if confidence is >= 0.75
for i in range(len(test_data)):
    confidence = max(probabilities[i])
    if confidence >= 0.75:
        predicted_label = "Yes" if predictions[i] == 1 else "No"
        print(f"{test_instances[i]}, {test_rows[i][1]}, {test_rows[i][2]}, {test_rows[i][3]}, {test_rows[i][4]}, {predicted_label}, {confidence:.3f}")
