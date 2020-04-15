from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Column names for reading in iris.csv
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = read_csv(r'C:\Users\thomaej\Documents\WGU\4th Semester\Machine Learning\Iris\iris.csv', names=names)

#Seperating my features from what I want to predict which is class
features = iris[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
labels = iris[['class']]

#Turning my dictionaries into numpy arrays
features_array = np.array(features)
label_array = np.array(labels)

#Looking at the data
print(iris.head())
iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Splitting the data into train and test sets, we will train on 75% of the data
feature_train, feature_test, label_train, label_test = train_test_split(features_array, label_array, train_size=.75, stratify=labels)

print(np.shape(feature_train))
print(np.shape(feature_test))
print(np.shape(label_train))
print(np.shape(label_test))

#Creating and predicting with a Decision Tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
print (accuracy_score(pred, label_test))
