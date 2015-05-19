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

from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from oracle import Oracle
from palpomdp import PALPOMDP
from paloriginal import *
from palbaseline import *


class Simulation(object):
    """ A class which executes a simulation of all algorithms on a dataset provided (via the file name). """

    def __init__(self, scenario, budget, clusterBudget, datasetFile, classIndex, trainSize, testSize, classifier, numIterations):
        """ The constructor for the simulation class.

            Parameters:
                scenario        --  The scenario, as a string, defining which group of oracles to use.
                budget          --  The budget available.
                clusterBudget   --  The budget for the clustering step. This is divided among all non-normal oracles.
                datasetFile     --  The dataset filename and path.
                classIndex      --  The class index in the column of the dataset matrix.
                trainSize       --  The number of train data points.
                testSize        --  The number of test data points.
                classifier      --  The classifier to use for the final test dataset with trained
                                    dataset labels taken from oracles.
                numIterations   --  The number of iterations to execute for evaluating performance.
        """

        self.B = budget
        self.Bc = clusterBudget
        self.scenario = scenario

        self.datasetFile = datasetFile
        self.classIndex = classIndex
        self.trainSize = trainSize
        self.testSize = testSize

        self.classifier = classifier
        self.numIterations = numIterations

        if self.scenario == 'original_1' or self.scenario == 'original_2' or self.scenario == 'original_3':
            self.numAlgorithms = 5
        elif self.scenario == 'baseline':
            self.numAlgorithms = 6
        else:
            self.numAlgorithms = 4

    def _initialize(self):
        """ Initialize a run of the simulation. This resets everything. """

        self._create_datasets()
        self._create_oracles()
        self._create_algorithms()

    def _create_datasets(self):
        """ Create the datasets by randomizing then splitting into sections based on the sizes provided. """

        # Load the dataset.
        self.dataset = np.genfromtxt(self.datasetFile, delimiter=',')
        self.numDataPoints = np.shape(self.dataset)[0]
        self.numClasses = len(set(self.dataset[:, self.classIndex]))

        self.labels = self.dataset[:, self.classIndex]
        nonClassIndexes = [i for i in range(self.dataset.shape[1]) if i != self.classIndex]
        self.dataset = self.dataset[:, nonClassIndexes]

        # Scalarize the dataset.
        self.dataset = preprocessing.scale(self.dataset)

        # Find the minimum number of data points within a class over all classes.
        minNumInClass = min([len({i for i in range(self.numDataPoints) if self.labels[i] == k}) for k in range(self.numClasses)])

        # Split the data based on classes, adding to one class only up to this minimum number. Do this
        # over a random ordering of the dataset to reduce sample bias.
        splitDataset = [list() for k in range(self.numClasses)]
        splitLabels = [list() for k in range(self.numClasses)]

        indexes = list(range(self.numDataPoints))
        rnd.shuffle(indexes)

        for k in range(self.numClasses):
            for i in indexes:
                if self.labels[i] == k:
                    splitDataset[k] += [self.dataset[i, :]]
                    splitLabels[k] += [self.labels[i]]

                if len(splitDataset[k]) >= minNumInClass:
                    break

        # Re-define the dataset to include an even split of each class.
        self.dataset = np.vstack([np.array(d) for d in splitDataset])
        self.labels = np.array(splitLabels, dtype=int).flatten()
        self.numDataPoints = self.dataset.shape[0]

        # Now, randomize this data again, guaranteeing that the *train* data has at least one index is selected
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

        self.trainIndexes = indexes[0:(self.trainSize - self.numClasses)] + requiredIndexes
        self.testIndexes = indexes[(self.trainSize - self.numClasses):(self.trainSize - self.numClasses + self.testSize)]

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

    def _create_oracles(self):
        """ Create the oracles, based on the scenario specified. """

        # Setup the oracles, one for each algorithm.
        if self.scenario == 'original_1':
            self.oracles = [[Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, reluctant=True)] \
                        for i in range(self.numAlgorithms)]
        elif self.scenario == 'original_2':
            self.oracles = [[Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, fallible=True)] \
                        for i in range(self.numAlgorithms)]
        elif self.scenario == 'original_3':
            self.oracles = [[Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, costVarying=True)] \
                        for i in range(self.numAlgorithms)]
        elif self.scenario == 'baseline':
            self.oracles = [[Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, reluctant=True),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, fallible=True),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, costVarying=True)] \
                        for i in range(self.numAlgorithms)]
        else:
            self.oracles = [[Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, reluctant=True),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, fallible=True),
                            Oracle(self.dataset, self.labels, self.classIndex, self.trainIndexes, costVarying=True)] \
                        for i in range(self.numAlgorithms)]

    def _create_algorithms(self):
        """ Create the algorithms, based on the scenario specified. """

        if self.scenario == 'original_1':
            self.algorithms = [PALPOMDP(self.Xtrain, self.numClasses, self.oracles[0], [0.0, self.Bc]),
                               PALOriginalScenario1(self.Xtrain, self.numClasses, self.oracles[1], [0.0, self.Bc]),
                               PALBaselineRandom(self.Xtrain, self.numClasses, self.oracles[2]),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[3], 0),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[4], 1)]
        elif self.scenario == 'original_2':
            self.algorithms = [PALPOMDP(self.Xtrain, self.numClasses, self.oracles[0], [0.0, self.Bc]),
                               PALOriginalScenario2(self.Xtrain, self.numClasses, self.oracles[1], [0.0, self.Bc]),
                               PALBaselineRandom(self.Xtrain, self.numClasses, self.oracles[2]),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[3], 0),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[4], 1)]
        elif self.scenario == 'original_3':
            self.algorithms = [PALPOMDP(self.Xtrain, self.numClasses, self.oracles[0], [0.0, self.Bc]),
                               PALOriginalScenario3(self.Xtrain, self.numClasses, self.oracles[1], [0.0, self.Bc]),
                               PALBaselineRandom(self.Xtrain, self.numClasses, self.oracles[2]),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[3], 0),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[4], 1)]
        elif self.scenario == 'baseline':
            self.algorithms = [PALPOMDP(self.Xtrain, self.numClasses, self.oracles[0], [0.0, self.Bc / 3.0, self.Bc / 3.0, self.Bc / 3.0]),
                               PALBaselineRandom(self.Xtrain, self.numClasses, self.oracles[1]),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[2], 0),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[3], 1),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[4], 2),
                               PALBaselineFixed(self.Xtrain, self.numClasses, self.oracles[5], 3)]
        else:
            self.algorithms = [PALPOMDP(self.Xtrain, self.numClasses, self.oracles[0], [0.0, self.Bc / 3.0, self.Bc / 3.0, self.Bc / 3.0]),
                               PALOriginalScenario1(self.Xtrain, self.numClasses, self.oracles[1], [0.0, self.Bc, 0.0, 0.0]),
                               PALOriginalScenario2(self.Xtrain, self.numClasses, self.oracles[2], [0.0, 0.0, self.Bc, 0.0]),
                               PALOriginalScenario3(self.Xtrain, self.numClasses, self.oracles[3], [0.0, 0.0, 0.0, self.Bc])]

        # Create and solve the PALPOMDP. Store the policy within the PALPOMDP class and use the custom
        # helper functions later to get the policy and update the belief. Try the same commands for the
        # other types, in case they need to do any computation.
        for algorithm in self.algorithms:
            algorithm.create()

    def execute(self):
        """ Execute the entire simulation. """

        accuracies = [list() for i in range(self.numAlgorithms)]
        costs = [list() for i in range(self.numAlgorithms)]
        #avgAccuracy = [0.0 for i in range(self.numAlgorithms)]
        #avgCost = [0.0 for i in range(self.numAlgorithms)]

        # For each iteration, we randomly re-split the dataset, re-create all algorithms and oracles,
        # execute each algorithm on the dataset, obtain labels, train a classifier using these
        # labels for the train set, then test the accuracy of the classifier on the test set. The
        # accuracy and cost results are averaged for each.
        for i in range(self.numIterations):
            #print(".", end='')
            print("Iteration %i of %i." % (i + 1, self.numIterations))

            # Re-create the split of train/test, the oracles, and algorithms.
            self._initialize()

            # For each algorithm, use the Xtrain to compute labels.
            for j, algorithm in enumerate(self.algorithms):
                print("Algorithm '%s' (%i of %i):" % (str(algorithm), j + 1, self.numAlgorithms))

                # The agent spent an inital cost for clustering.
                currentCost = algorithm.pal.Ctotal

                # If the algorithm ever runs out of the budget, stop.
                while currentCost < self.B:
                    # The algorithm selects an oracle and the data point (Xtrain dataset index).
                    action, dataPointIndex = algorithm.select()

                    if action is None:
                        break

                    # We query that oracle.
                    label, cost = self.oracles[j][action].query(dataPointIndex)

                    currentCost += cost

                    # The algorithm updates internal information.
                    algorithm.update(label, cost)

                    print("\tAction: %i\t DP Index: %s\t Label: %s\t Cost: %.3f\t Spent: %.3f\t Budget: %.3f" % (action, str(dataPointIndex), str(label), cost, currentCost, self.B))

                # Perform necessary finishing tasks with this algorithm, then get the
                # proactively learned labels and dataset.
                dataset, labels = algorithm.finish()

                #print("Final Algorithm Constructed Dataset:")
                #print(dataset)
                #print(labels)

                # Train using k-nearest neighbors, logistic regression, or an SVM.
                c = None
                if self.classifier == 'knn':
                    c = KNeighborsClassifier(5)
                elif self.classifier == 'logistic_regression':
                    c = LogisticRegression(C=1e5)
                elif self.classifier == 'svm':
                    c = SVC(kernel='rbf', max_iter=1000)
                else:
                    print("Error: Invalid classifier '%s'." % (self.classifier))
                    raise Exception()

                accuracy = 0.0
                #try:
                # Attempt to learn a model. This may fail, e.g., the number of data points is
                # less than the number of features. This might happen for the baseline oracles,
                # such as the one who constantly picks the oracle who is reluctant to answer.
                # If it fails, then the accuracy is zero.
                c.fit(dataset, labels)

                # Predict for each of the data points in the test set.
                yprediction = c.predict(self.Xtest)

                # Compute accuracy!
                #accuracy = np.sum(self.ytest == yprediction) / self.ytest.size
                accuracy = metrics.accuracy_score(self.ytest, yprediction)
                #except ValueError:
                #    pass

                # Update accuracy and cost!
                accuracies[j] += [accuracy]
                costs[j] += [currentCost]
                #avgAccuracy[j] = (float(i) * avgAccuracy[j] + accuracy) / float(i + 1)
                #avgCost[j] = (float(i) * avgCost[j] + currentCost) / float(i + 1)

                print("\tCost:      %.3f" % (currentCost))
                print("\tAccuracy:  %.3f" % (accuracy))

            sys.stdout.flush()

        #print(" Done")

        #for j, algorithm in enumerate(self.algorithms):
        #    print("Algorithm '%s' (%i of %i):" % (str(algorithm), j + 1, self.numAlgorithms))
        #    print("\tTotal Average Cost:      %.3f +/- %.3f" % (np.mean(costs[j]), np.std(costs[j])))
        #    print("\tTotal Average Accuracy:  %.3f +/- %.3f" % (np.mean(accuracies[j]), np.std(accuracies[j])))
        #    #print("\tTotal Average Cost:      %.3f" % (avgCost[j]))
        #    #print("\tTotal Average Accuracy:  %.3f" % (avgAccuracy[j]))

        return [str(algorithm) for algorithm in self.algorithms], accuracies, costs #avgCost, avgAccuracy

if __name__ == "__main__":
    #try:
        print("Performing Simulation...")

        sim = Simulation(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]),
                        int(sys.argv[6]), int(sys.argv[7]), sys.argv[8], int(sys.argv[9]))
        algorithmNames, accuracies, costs = sim.execute()

        for j, name in enumerate(algorithmNames):
            print("Algorithm '%s' (%i of %i):" % (name, j + 1, len(algorithmNames)))
            print("\tTotal Average Cost:      %.3f +/- %.3f" % (np.mean(costs[j]), np.std(costs[j])))
            print("\tTotal Average Accuracy:  %.3f +/- %.3f" % (np.mean(accuracies[j]), np.std(accuracies[j])))

        print("Done.")
    #except Exception:
    #    print("Syntax:   python simulation.py <scenario> <budget> <cluster budget> <dataset> <class index> <train size> <test size> <classifier> <num iterations>")

