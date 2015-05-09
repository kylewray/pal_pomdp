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
import itertools as it

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
        self.PrCorrect = None # Pr(correct | data point, oracle) as 2-d array.
        self.PrAnswer = None # Pr(answer | data point, oracle) as 2-d array.
        self.CostOracle = None # C(data point, oracle) as 2-d array.

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

            # Note: The number of 'features' in the file: Pr(correct), Pr(answer), Cost(oracle).
            numFeatures = 3

            rowOffset = 1
            self.PrCorrect = np.array([[float(data[tdp + rowOffset][o * numFeatures + 0]) \
                                        for o in range(self.numOracles)] \
                                    for tdp in range(self.trainSize)])
            self.PrAnswer = np.array([[float(data[tdp + rowOffset][o * numFeatures + 1]) \
                                        for o in range(self.numOracles)] \
                                    for tdp in range(self.trainSize)])
            self.CostOracle = np.array([[float(data[tdp + rowOffset][o * numFeatures + 2]) \
                                        for o in range(self.numOracles)] \
                                    for tdp in range(self.trainSize)])
        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def create(self):
        """ Create the POMDP once the oracles and their probabilities have been defined. """

        # Create the states.
        states = list() #['Initial'] #, 'Success', 'Failure']
        for i in range(self.trainSize):
            # For this data point (i.e., dataset[i, :]) create the corresponding states as a tuple:
            # < num correct, num incorrect, oracle responded >
            for numCorrect in range(i + 1):
                states += [(numCorrect, i - numCorrect, True), (numCorrect, i - numCorrect, False)]

        # The last set of states for the last data point only has true, which self-loops, since we are done.
        for numCorrect in range(self.trainSize + 1):
            states += [(numCorrect, self.trainSize - numCorrect, True)]

        self.n = len(states)

        # Create the actions.
        actions = [i for i in range(self.numOracles)]
        self.m = len(actions)

        # Create the state transitions.
        self.T = [[[0.0 for sp in range(self.n)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(states):
            for a, action in enumerate(actions):
                for sp, statePrime in enumerate(states):
                    #print(state, action, statePrime)

                    if state[0] + state[1] == self.trainSize and state == statePrime:
                        self.T[s][a][sp] = 1.0

                    elif state[0] == statePrime[0] and state[1] == statePrime[1] and \
                            ((state[2] == True and statePrime[2] == False) or \
                            (state[2] == False and statePrime[2] == False)):
                        # If no transition occurred and this was the first time the oracle did not respond,
                        # then we assign the probability of no answer (response) which equals
                        # 1 - Pr(answer | oracle, data point).
                        dataPointIndex = state[0] + state[1]
                        self.T[s][a][sp] = 1.0 - self.PrAnswer[dataPointIndex, action]

                    elif (state[0] + 1 == statePrime[0] and state[1] == statePrime[1]) and \
                            ((state[2] == True and statePrime[2] == True) or \
                            (state[2] == False and statePrime[2] == True)):
                        # If a successful labeling occurred, meaning that only the number correct increased,
                        # then we assign the probability of a correct labeling for this oracle, which equals
                        # Pr(answer | oracle, data point of s [not sp]) * Pr(correct | oracle, data point)
                        dataPointIndex = statePrime[0] + statePrime[1] - 1
                        self.T[s][a][sp] = self.PrAnswer[dataPointIndex - 1, action] * self.PrCorrect[dataPointIndex, action]

                    elif (state[0] == statePrime[0] and state[1] + 1 == statePrime[1]) and \
                            ((state[2] == True and statePrime[2] == True) or \
                            (state[2] == False and statePrime[2] == True)):
                        # If a failure of labeling occurred, meaning that only the number incorrect increased,
                        # then we assign the probability of an incorrect labeling for this oracle, which equals
                        # Pr(answer | oracle, data point of s [not sp]) * (1 - Pr(correct | oracle, data point))
                        dataPointIndex = statePrime[0] + statePrime[1] - 1
                        self.T[s][a][sp] = self.PrAnswer[dataPointIndex, action] * (1.0 - self.PrCorrect[dataPointIndex, action])

                #print(sum([self.T[s][a][sp] for sp in range(len(states))]))
                    
        self.T = np.array(self.T)

        # Create the observations.
        observations = [True, False]
        self.z = len(observations)

        # Create the observation transitions.
        self.O = [[[0.0 for o in range(self.z)] for sp in range(self.n)] for a in range(self.z)]
        for a, action in enumerate(actions):
            for sp, statePrime in enumerate(states):
                for o, observation in enumerate(observations):
                    if (statePrime[2] == True and observation == True) or (statePrime[2] == False and observation == False):
                        self.O[a][sp][o] = 1.0

                #print(sum([self.O[a][sp][o] for o in range(len(observations))]))

        self.O = np.array(self.O)

        # Create the belief points.
        self.B = list()

        # Method #1: Simple. Add the pure belief for each state.
        for s, state in enumerate(states):
            b = [0.0 for i in range(self.n)]
            b[s] = 1.0
            self.B += [b]

        # Method #2: Add the mixed belief among each collection of states which are "probabilistically entangled."
        #for i in range(self.trainSize):
        #    numNonZeroBelief = float(i + 1)

        #    for answered in [True, False]:
        #        for numCorrect in range(i + 1):
        #            s = states.index((numCorrect, i - numCorrect, answered))

        # Important: We do *not* need to add belief points for the final set of states; these states have
        # no meaningful action to take there, since they are all absorbing with zero reward.
        #for numCorrect in range(self.trainSize + 1):
        #    states += [(numCorrect, self.trainSize - numCorrect, True)]

        self.r = len(self.B)
        self.B = np.array(self.B)

        # Create the reward functions.
        self.k = 1
        self.R = [[[0.0 for a in range(self.m)] for s in range(self.n)] for i in range(self.k)]
        for s, state in enumerate(states):
            # Note: You are currently at (s[0] + s[1] - 1), but since we pay an action to label the
            # *next* data point, we are really looking at paying the dataPointIndex + 1.
            dataPointIndex = (state[0] + state[1] - 1) + 1

            # If this was the last data point row (which are absorbing states), then no cost.
            if dataPointIndex == self.trainSize:
                continue

            for a, action in enumerate(actions):
                self.R[0][s][a] = -self.CostOracle[dataPointIndex, a]

        self.R = np.array(self.R)

        self.horizon = self.trainSize

        self._compute_optimization_variables()

    def _max_alpha_vector(self, b, Gamma, pi):
        """ Return the best value and action at the belief point given.

            Parameters:
                b       --  The belief point (numpy array).
                Gamma   --  The alpha-vectors for the policy (numpy array).
                pi      --  The corresponding actions for each alpha-vector of the policy (numpy array).
        """

        v = np.matrix(Gamma) * np.matrix(b).T
        return np.max(v), np.argmax(v)

    def simulate(self, Gamma, pi, outputHistory=False):
        """ Simulate the execution of a given policy. Optionally output the action-observation history. This
            stores the history for use in creating a final classifier.

            Parameters:
                Gamma           --  The alpha-vectors for the policy.
                pi              --  The corresponding actions for each alpha-vector of the policy.
                outputHistory   --  Optionally output the action-observation history. Default is False.
        """

        # We know the true state of the system initially.
        b = np.array([0.0 for s in range(self.n)])
        b[0] = 1.0
        s = 0

        for t in range(self.horizon):
            # Take the action.
            v, a = self._max_alpha_vector(b, Gamma, pi)

            print("Action: %i" % (a))

            # Stochastically transition the state.
            sp = None
            target = rnd.random()
            current = 0.0
            for spIter in range(self.n):
                current += self.T[s, a, spIter]
                if current >= target:
                    sp = spIter
                    break

            # Stochastically make an observation.
            o = None
            target = rnd.random()
            current = 0.0
            for oIter in range(self.z):
                current += self.O[a, sp, oIter]
                if current >= target:
                    o = oIter
                    break

            # Update the belief point.
            bNew = np.array([self.O[a, spIter, o] * np.dot(self.T[:, a, spIter], b) for spIter in range(self.n)])
            b = bNew / bNew.sum()

            # Transition the state!
            s = sp

            print(b)

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

