import sys
import random as rnd
import numpy as np

from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


fullDataset = np.genfromtxt(sys.argv[1], delimiter=',')
numDataPoints = int(fullDataset.shape[0])
classIndex = int(sys.argv[2])
trainSize = int(sys.argv[3])
testSize = int(sys.argv[4])

indexes = list(range(numDataPoints))
rnd.shuffle(indexes)
fullDataset = fullDataset[indexes, :]

dataset = fullDataset[:, [i for i in range(fullDataset.shape[1]) if i != classIndex]]
labels = fullDataset[:, classIndex]

dataset = preprocessing.scale(dataset)

Xtrain = dataset[:trainSize, :]
ytrain = labels[:trainSize]

#classifier = KNeighborsClassifier(5)
classifier = SVC(kernel='rbf', max_iter=10000)

classifier.fit(Xtrain, ytrain)

Xtest = dataset[trainSize:(trainSize + testSize), :]
ytest = labels[trainSize:(trainSize + testSize)]

yprediction = classifier.predict(Xtest)

#accuracy = np.sum(ytest == yprediction) / ytest.size
accuracy = metrics.accuracy_score(ytest, yprediction)

print("Train Size:", Xtrain.shape[0])
print("Test Size:", Xtest.shape[0])
print("Accuracy:", accuracy)

