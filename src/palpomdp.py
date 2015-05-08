""" The MIT License (MIT)

    Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "nova", "wrapper"))
from nova.pomdp import *

import csv
import numpy as np
import random as rnd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA


class PALPOMDP(MOPOMDP):
    """ A class which contains oracles. """

    def __init__(self):
        """ The constructor of the PAL POMDP class. """

        super().__init__()

        # The dataset: number of data points, number of classes, and actual data.
        self.numDataPoints = 0
        self.numClasses = 0
        self.trainSize = 0
        self.testSize = 0
        self.kNCC = 0
        self.dataset = None
        self.classIndex = 0

        # The oracle information.
        self.numOracles = 0
        self.responsePr = None # Pr(response class | oracle, true class)
        self.oracleCost = None # C(oracle, data point)

    def load(self, filename, shuffleDataset=True):
        """ Load a PAL file, containing the oracle and dataset information.

            Parameters:
                filename        --  The name and path of the file to load.
                suffleDataset   --  Optionally shuffle the dataset when it is loaded. Default is True.
        """

        # Load all the data into this object.
        data = list()
        filePath = None
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

            # Remember the path to this file, since it will be used as a base for
            # loading the data file.
            filePath = os.path.dirname(os.path.realpath(filename))

        # Attempt to parse all the data into their respective variables.
        try:
            # Load the variables corresponding to the dataset.
            datasetFilename = os.path.join(filePath, str(data[0][1]))
            self.dataset = np.genfromtxt(datasetFilename, delimiter=',')
            self.numDataPoints = np.shape(self.dataset)[0]
            self.classIndex = int(data[0][2])
            self.numClasses = len(set(self.dataset[:, self.classIndex]))

            if shuffleDataset:
                indexes = list(range(self.numDataPoints))
                rnd.shuffle(indexes)
                self.dataset = self.dataset[indexes, :]

            # Load the variables corresponding to the training/testing classifier.
            self.classifier = str(data[0][3])
            self.trainSize = int(data[0][4])
            self.testSize = int(data[0][5])
            self.kNCC = int(data[0][6])

            # Load the oracle information.
            self.numOracles = int(data[0][0])

            rowOffset = 1
            self.responsePr = np.array([[[float(data[(self.numClasses * o + tc) + rowOffset][rc]) \
                                        for rc in range(self.numClasses)] \
                                    for o in range(self.numOracles)] \
                                for tc in range(self.numClasses)])

            rowOffset = 1 + self.numOracles * self.numClasses
            self.oracleCost = np.array([[float(data[tdp + rowOffset][o]) \
                                    for o in range(self.numOracles)] \
                                for tdp in range(self.trainSize)])
        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def create(self):
        """ Create the POMDP once the oracles and their probabilities have been defined. """

        # First, create the set of states.
        self.n = 0

        #self._compute_optimization_variables()

    def simulate(self, outputHistory=False):
        """ Simulate the execution of a policy. Optionally output the action-observation history. This
            stores the history for use in creating a final classifier.

            Parameters:
                outputHistory   --  Optionally output the action-observation history. Default is False.
        """

        pass

    def train(self):
        """ Train the final classifier after a simulation has generated a history of actions and observations. """

        # Split apart the features and class labels.
        nonClassIndexes = [i for i in range(self.dataset.shape[1]) if i != self.classIndex]
        dataset = self.dataset[:, nonClassIndexes]
        labels = self.dataset[:, self.classIndex]

        # Create the training subset of the data (the first 500 values.)
        Xtrain = dataset[:self.trainSize]
        ytrain = labels[:self.trainSize]

        # Train using k-nearest neighbors.
        knn = KNeighborsClassifier(5)
        knn.fit(Xtrain, ytrain)

        # Create the test subset of the data (the values from 501 onwards).
        Xtest = dataset[self.trainSize:(self.trainSize + self.testSize)]
        ytest = labels[self.trainSize:(self.trainSize + self.testSize)]

        # Predict for each of these.
        yprediction = knn.predict(Xtest[:])

        # Compute accuracy!
        accuracy = np.sum(ytest == yprediction) / ytest.size
        print("K-Nearest Neighbors Accuracy:", accuracy)

        #classifiers = [KNeighborsClassifier(3),
        #                KNeighborsClassifier(5),
        #                SVC(kernel="linear", max_iter=1000),
        #                SVC(kernel="poly", max_iter=1000),
        #                SVC(kernel="rbf", max_iter=1000),
        #                DecisionTreeClassifier(max_depth=5),
        #                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #                GaussianNB(),
        #                LDA(),
        #                QDA()]


    def test(self):
        """ Test the accuracy of the final classifier once it has been trained. """

        pass

