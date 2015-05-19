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

import random as rnd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


ORACLE_DEFAULT_COST = 0.5
ORACLE_DEFAULT_COST_RATIO = 0.25
#ORACLE_MIN_COST = ORACLE_DEFAULT_COST * ORACLE_DEFAULT_COST_RATIO * ORACLE_DEFAULT_COST_RATIO
ORACLE_DATA_SUBSET = 0.01
ORACLE_UNCERTAINTY_THRESHOLD = 0.75


class Oracle(object):
    """ An oracle class which has the *true* probabilities, and is able to simulate them given a data point. """

    def __init__(self, dataset, labels, classIndex, mapping, reluctant=False, fallible=False, costVarying=False,
                dataSubset=ORACLE_DATA_SUBSET, uncertaintyThreshold=ORACLE_UNCERTAINTY_THRESHOLD,
                defaultCost=ORACLE_DEFAULT_COST, costRatio=ORACLE_DEFAULT_COST_RATIO):
        """ The constructor of the Oracle.

            Parameters:
                dataset                 --  The true values in the dataset from the simulation itself.
                labels                  --  The true labels from the dataset from the simulation itself.
                classIndex              --  The class index within the dataset matrix.
                mapping                 --  Maps indexes of Xtrain to those of the true dataset.
                reluctant               --  This oracle varies in its ability to answer. Default is False.
                fallible                --  This oracle varies in its ability to answer correctly. Default
                                            is False.
                costVarying             --  This oracle varies its cost. Default is False.
                dataSubset              --  The subset ratio of the dataset for determining probabilities. Default
                                            is ORACLE_DATA_SUBSET.
                uncertaintyThreshold    --  The uncertainty threshold for determining indexes. Default is
                                            ORACLE_UNCERTAINTY_THRESHOLD.
                defaultCost             --  The default cost of the oracle. Default is ORACLE_DEFAULT_COST.
                costRatio               --  The ratio reducing cost for each used in the set {reluctant, fallible}.
                                            Default is ORACLE_DEFAULT_COST_RATIO.
        """

        self.mapping = list(mapping)

        # Load the dataset.
        #self.dataset = np.genfromtxt(datasetFile, delimiter=',')
        #self.numDataPoints = np.shape(self.dataset)[0]
        #self.numClasses = len(set(self.dataset[:, classIndex]))

        #self.labels = self.dataset[:, classIndex]
        #nonClassIndexes = [i for i in range(self.dataset.shape[1]) if i != classIndex]
        #self.dataset = self.dataset[:, nonClassIndexes]

        # Scalarize the dataset.
        #self.dataset = preprocessing.scale(self.dataset)

        self.dataset = dataset.copy()
        self.labels = labels.copy()

        self.numDataPoints = self.dataset.shape[0]
        self.numClasses = len(set(self.labels))

        # Handle each of the special traits that make the oracle interesting.
        cost = defaultCost
        if reluctant:
            self.failToAnswer = self._random_train(dataSubset, uncertaintyThreshold)
            cost *= costRatio
        else:
            self.failToAnswer = None

        self.reluctant = reluctant

        if fallible:
            self.randomClass = self._random_train(dataSubset, uncertaintyThreshold)
            cost *= costRatio
        else:
            self.randomClass = None

        self.fallible = fallible

        self.costs = np.array([cost for i in range(self.numDataPoints)])
        if costVarying:
            self._compute_varied_costs(dataSubset, defaultCost, costRatio)

        self.costVarying = costVarying

    def _random_train(self, dataSubset, uncertaintyThreshold):
        """ Randomly train on a subset of the data, and find indexes within the specified uncertainty probability range.

            Parameters:
                dataSubset              --  The proportion of the dataset to randomly train on, prior to returning the indexes.
                uncertaintyThreshold    --  The uncertainty threshold for determining indexes.

            Returns:
                A set of indexes for which 
        """

        # Select the random subset on which this oracle is trained. Also, guarantee that at least one index is selected
        # from each of the classes.
        indexes = list(range(self.numDataPoints))
        rnd.shuffle(indexes)

        requiredIndexes = list()
        for c in range(self.numClasses):
            for i in indexes:
                if int(self.labels[i]) == c:
                    requiredIndexes += [i]
                    break

        indexes = list(set(indexes[0:max(self.numClasses, int(dataSubset * self.numDataPoints))]) | set(requiredIndexes))

        # Train the oracle. Then predict the probabilities over the entire dataset.
        c = LogisticRegression(C=1e5)
        c.fit(self.dataset[indexes, :], self.labels[indexes].astype(int))
        Pr = c.predict_proba(self.dataset)

        # Return the set of indexes, over the entire dataset, which have a predicted probability
        # within the threshold.
        return [i for i in range(self.numDataPoints) if max([Pr[i, j] for j in range(self.numClasses)]) < uncertaintyThreshold]

    def _compute_varied_costs(self, dataSubset, defaultCost, costRatio):
        """ Compute the varied costs as a function of the data points.

            Parameters:
                dataSubset      --  The proportion of the dataset to randomly train on, prior to returning the indexes.
                defaultCost     --  The default cost of the oracle.
                costRatio       --  The ratio reducing cost for each used in the set {reluctant, fallible}.
        """

        # Select the random subset on which this oracle is trained.
        indexes = list(range(self.numDataPoints))
        rnd.shuffle(indexes)

        requiredIndexes = list()
        for c in range(self.numClasses):
            for i in indexes:
                if int(self.labels[i]) == c:
                    requiredIndexes += [i]
                    break

        indexes = list(set(indexes[0:max(self.numClasses, int(dataSubset * self.numDataPoints))]) | set(requiredIndexes))

        # Train the oracle. Then predict the probabilities over the entire dataset.
        c = LogisticRegression(C=1e5)
        c.fit(self.dataset[indexes, :], self.labels[indexes])
        Pr = c.predict_proba(self.dataset)

        # Return the set of indexes, over the entire dataset, which have a predicted probability
        # within the threshold.
        #self.costs = np.array([max(ORACLE_MIN_COST, 1.0 - (max([Pr[i, j] for j in range(self.numClasses)]) - \
        #                                1.0 / self.numClasses) / (1.0 - 1.0 / self.numClasses)) \
        #                            for i in range(self.numDataPoints)])

        oracleMinCost = defaultCost * costRatio * costRatio

        self.costs = np.array([1.0 - (max([Pr[i, j] for j in range(self.numClasses)]) - \
                                        1.0 / self.numClasses) / (1.0 - 1.0 / self.numClasses) + oracleMinCost \
                                    for i in range(self.numDataPoints)])

        #print("\n".join(["%.3f" % (c) for c in self.costs]))

    def query(self, dataPointIndex):
        """ Return the oracle's response to the data point, as well as the corresponding cost.

            Parameters:
                dataPointIndex  --  The index of the data point within the Xtrain dataset.

            Returns:
                label   --  The label in {0, .., numClasses}, or None if no response.
                cost    --  The cost of this query.
        """

        dataPointIndex = self.mapping[dataPointIndex]

        # First, see if the oracle answers at all based on the data point being in the 'blacklist'.
        if self.failToAnswer is not None and dataPointIndex in self.failToAnswer:
            return None, self.costs[dataPointIndex]

        # Now, the oracle responds; however, if the data point is in the randomClass list, then
        # randomly select a class label instead.
        if self.randomClass is not None and dataPointIndex in self.randomClass:
            return rnd.randint(0, self.numClasses - 1), self.costs[dataPointIndex]

        # If the function made it here, then return the correct label and the cost.
        return int(self.labels[dataPointIndex]), self.costs[dataPointIndex]

    def get_cost(self, dataPointIndex):
        """ Return how much a particular data point will cost to label.

            Parameters:
                dataPointIndex  --  The index of the data point within the Xtrain dataset.

            Returns:
                The cost of the query.
        """

        dataPointIndex = self.mapping[dataPointIndex]
        return self.costs[dataPointIndex]


if __name__ == "__main__":
    print("Performing Oracle Unit Test...")

    classIndex = 4
    dataset = np.genfromtxt("../experiments/iris/iris.data", delimiter=',')

    labels = dataset[:, classIndex]
    nonClassIndexes = [i for i in range(dataset.shape[1]) if i != classIndex]
    dataset = dataset[:, nonClassIndexes]

    dataset = preprocessing.scale(dataset)

    mapping = range(150)

    oracles = [Oracle(dataset, labels, classIndex, mapping),
                Oracle(dataset, labels, classIndex, mapping, reluctant=True),
                Oracle(dataset, labels, classIndex, mapping, fallible=True),
                Oracle(dataset, labels, classIndex, mapping, costVarying=True)]

    for i in range(150):
        for oi, o in enumerate(oracles):
            y, c = o.query(i)
            print("Oracle %i: %s\t%.3f\t\t" % (oi + 1, str(y), c), end='')
        print()

    print("Done.")

