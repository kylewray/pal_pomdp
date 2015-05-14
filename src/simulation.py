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

import sys

import numpy as np
import random as rnd

from oracle import Oracle
from palpomdp import PALPOMDP
from paloriginal import PALOriginal


class Simulation(object):
    """ A class which executes a simulation of all algorithms on a dataset provided (via the file name). """

    def __init__(self, scenario, datasetFile, classIndex, trainSize, testSize, classifier, kncc):
        """ The constructor for the simulation class.

            Parameters:
                scenario        --  The scenario, as a string, defining which group of oracles to use.
                datasetFile     --  The dataset filename and path.
                classIndex      --  The class index in the column of the dataset matrix.
                trainSize       --  The number of train data points.
                testSize        --  The number of test data points.
                classifier      --  The classifier to use for the final k-fold normalized cross-validation.
                kncc            --  The number of folds in k-fold normalized cross-validation.
        """

        self.algorithms = [PALPOMDP(), PALOriginal()]
        self.numAlgorithms = len(self.algorithms)

        self.classifier = classifier
        self.kncc = kncc

        self.finalAccuracy = [0.0 for i in range(self.numAlgorithms)]
        self.finalCosts = [0.0 for i in range(self.numAlgorithms)]

        self._create_datasets(datasetFile, classIndex, trainSize, testSize)
        self._create_oracles(scenario, datasetFile, classIndex)

    def _create_datasets(self, datasetFile, classIndex, trainSize, testSize):
        """ Create the datasets by randomizing then splitting into sections based on the sizes provided.

            Parameters:
                datasetFile     --  The dataset filename and path.
                classIndex      --  The class index in the column of the dataset matrix.
                trainSize       --  The number of train data points.
                testSize        --  The number of test data points.
        """

        # Load the dataset.
        self.dataset = np.genfromtxt(datasetFile, delimiter=',')
        self.numDataPoints = np.shape(self.dataset)[0]
        self.numClasses = len(set(self.dataset[:, classIndex]))

        self.labels = self.dataset[:, classIndex]
        nonClassIndexes = [i for i in range(self.dataset.shape[1]) if i != classIndex]
        self.dataset = self.dataset[:, nonClassIndexes]

        # Randomize the data, guaranteeing that the train data has at least one index is selected
        # from each of the classes.
        indexes = list(range(self.numDataPoints))
        rnd.shuffle(indexes)

        requiredIndexes = list()
        for c in range(self.numClasses):
            for i in indexes:
                if int(self.labels[i]) == c:
                    requiredIndexes += [i]
                    break
        indexes = list(set(indexes) - set(requiredIndexes))
        rnd.shuffle(indexes)

        self.trainIndexes = indexes[0:(trainSize - self.numClasses)] + requiredIndexes
        self.testIndexes = indexes[(trainSize - self.numClasses):(trainSize - self.numClasses + testSize)]

        #print(self.trainIndexes)
        #print(self.testIndexes)

        # Split the dataset into train and test.
        self.Xtrain = self.dataset[self.trainIndexes, :]
        self.ytrain = self.labels[self.trainIndexes]
        self.Xtest = self.dataset[self.testIndexes, :]
        self.ytest = self.labels[self.testIndexes]

        #print(self.Xtrain)
        #print(self.ytrain)
        #print(self.Xtest)
        #print(self.ytest)

    def _create_oracles(self, scenario, datasetFile, classIndex):
        """ Create the oracles, based on the scenario specified.

            Parameters:
                scenario        --  The scenario, as a string, defining which group of oracles to use.
                datasetFile     --  The dataset filename and path.
                classIndex      --  The class index in the column of the dataset matrix.
        """

        self.paymentCallback = [lambda c: self._payment(i, c) for i in range(self.numAlgorithms)]

        # Setup the oracles, one for each algorithm.
        if scenario == 'original_1':
            self.oracles = [[Oracle(datasetFile, classIndex, self.paymentCallback[i]),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], reluctant=True)] \
                        for i in range(self.numAlgorithms)]
        elif scenario == 'original_2':
            self.oracles = [[Oracle(datasetFile, classIndex, self.paymentCallback[i]),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], fallible=True)] \
                        for i in range(self.numAlgorithms)]
        elif scenario == 'original_3':
            self.oracles = [[Oracle(datasetFile, classIndex, self.paymentCallback[i]),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], variedCosts=True)] \
                        for i in range(self.numAlgorithms)]
        else:
            self.oracles = [[Oracle(datasetFile, classIndex, self.paymentCallback[i]),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], reluctant=True),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], fallible=True),
                            Oracle(datasetFile, classIndex, self.paymentCallback[i], variedCosts=True)] \
                        for i in range(self.numAlgorithms)]

    def _payment(self, algorithmIndex, cost):
        """ Execute a payment for the algorithm by the amount provided.

            Parameters:
                algorithmIndex  --  The index of the algorithm.
                cost            --  The cost of the payment.
        """

        self.finalCosts[algorithmIndex] += cost

    def execute(self):
        """ Execute the entire simulation. """

        pass


if __name__ == "__main__":
    #try:
        print("Performing Simulation...")

        sim = Simulation(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                        int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]))

        print("Done.")
    #except Exception:
    #    print("Syntax:   python simulation.py <scenario> <dataset> <class index> <train size> <test size> <ncc classifier> <ncc k>")

