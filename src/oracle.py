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
                oracleType='unknown', PrAnswerRange=None, PrCorrectRange=None, CostRange=None,
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
                oracleType              --  The probability type of oracle: 'known' or 'unknown'. Default is 'unknown'.
                PrAnswerRange           --  The range (min, max) for the random number generator for Pr(answer|x).
                                            If assigned None, then it is 1.0 for all data points. Default is None.
                PrCorrectRange          --  The range (min, max) for the random number generator for Pr(correct|x).
                                            If assigned None, then it is 1.0 for all data points. Default is None.
                CostRange               --  The range (min, max) for the random number generator for C(x).
                                            If assigned None, then it is the fixed cost for all data points. Default is None.
                dataSubset              --  The subset ratio of the dataset for determining probabilities. Default
                                            is ORACLE_DATA_SUBSET.
                uncertaintyThreshold    --  The uncertainty threshold for determining indexes. Default is
                                            ORACLE_UNCERTAINTY_THRESHOLD.
                defaultCost             --  The default cost of the oracle. Default is ORACLE_DEFAULT_COST.
                costRatio               --  The ratio reducing cost for each used in the set {reluctant, fallible}.
                                            Default is ORACLE_DEFAULT_COST_RATIO.
        """

        self.oracleType = oracleType
        self.mapping = list(mapping)

        self.dataset = dataset.copy()
        self.labels = labels.copy()

        self.numDataPoints = self.dataset.shape[0]
        self.numClasses = len(set(self.labels))

        # Handle each of the special traits that make the oracle interesting.
        cost = defaultCost
        if reluctant:
            if self.oracleType == 'unknown':
                self.failToAnswer = self._random_train(dataSubset, uncertaintyThreshold)
            else:
                self.PrAnswer = np.array([rnd.uniform(PrAnswerRange[0], PrAnswerRange[1]) for i in range(self.numDataPoints)])
            cost *= costRatio
        else:
            self.failToAnswer = None
            self.PrAnswer = None

        self.reluctant = reluctant

        if fallible:
            if self.oracleType == 'unknown':
                self.randomClass = self._random_train(dataSubset, uncertaintyThreshold)
            else:
                self.PrCorrect = np.array([rnd.uniform(PrCorrectRange[0], PrCorrectRange[1]) for i in range(self.numDataPoints)])
            cost *= costRatio
        else:
            self.randomClass = None
            self.PrCorrect = None

        self.fallible = fallible

        self.costs = np.array([cost for i in range(self.numDataPoints)])
        if costVarying:
            if self.oracleType == 'unknown':
                self._compute_varied_costs(dataSubset, defaultCost, costRatio)
            else:
                if CostRange is not None:
                    self.costs = np.array([rnd.uniform(CostRange[0], CostRange[1]) for i in range(self.numDataPoints)])
                else:
                    self.costs = np.array([cost for i in range(self.numDataPoints)])

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

        # Depending on the type of oracle, use a different calculation for the query.
        if self.oracleType == 'unknown':
            # First, see if the oracle answers at all based on the data point being in the 'blacklist'.
            if self.failToAnswer is not None and dataPointIndex in self.failToAnswer:
                return None, self.costs[dataPointIndex]

            # Now, the oracle responds; however, if the data point is in the randomClass list, then
            # randomly select a class label instead.
            if self.randomClass is not None and dataPointIndex in self.randomClass:
                return rnd.randint(0, self.numClasses - 1), self.costs[dataPointIndex]

            # If the function made it here, then return the correct label and the cost.
            return int(self.labels[dataPointIndex]), self.costs[dataPointIndex]
        else:
            # If a random number yields that the oracle will not answer, then do not but still pay the cost.
            if self.PrAnswer is not None and rnd.random() >= self.PrAnswer[dataPointIndex]:
                return None, self.costs[dataPointIndex]

            # Now, the oracle responds; however, if a random number yields that the oracle will be incorrect,
            # then randomly pick one of the *other* classes and pay the cost.
            if self.PrCorrect is not None and rnd.random() >= self.PrCorrect[dataPointIndex]:
                wrongClasses = [i for i in range(self.numClasses) if i != int(self.labels[dataPointIndex])]
                return rnd.choice(wrongClasses), self.costs[dataPointIndex]

            # If the function made it here, then return the correct label and the cost.
            return int(self.labels[dataPointIndex]), self.costs[dataPointIndex]

    def get_type(self):
        """ Return the type of this oracle: 'known' or 'unknown'.

            Returns:
                The type of the oracle.
        """

        return self.oracleType

    def get_pr_answer(self, dataPointIndex):
        """ Return the probability that a particular data point will be answered. This requires the oracle to be of type 'known'.

            Parameters:
                dataPointIndex  --  The index of the data point within the Xtrain dataset.

            Returns:
                The probability the oracle will respond, or None if this is not of type 'known'.
        """

        if self.oracleType == 'known':
            if self.PrAnswer is not None:
                dataPointIndex = self.mapping[dataPointIndex]
                return self.PrAnswer[dataPointIndex]
            else:
                return 1.0
        else:
            return None

    def get_pr_correct(self, dataPointIndex):
        """ Return the probability that a particular data point will be correct. This requires the oracle to be of type 'known'.

            Parameters:
                dataPointIndex  --  The index of the data point within the Xtrain dataset.

            Returns:
                The probability the oracle will be correct, or None if this is not of type 'known'.
        """

        if self.oracleType == 'known':
            if self.PrCorrect is not None:
                dataPointIndex = self.mapping[dataPointIndex]
                return self.PrCorrect[dataPointIndex]
            else:
                return 1.0
        else:
            return None

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
                Oracle(dataset, labels, classIndex, mapping, reluctant=True,
                        oracleType='known', PrAnswerRange=[0.25, 1.0]),
                Oracle(dataset, labels, classIndex, mapping, fallible=True),
                Oracle(dataset, labels, classIndex, mapping, fallible=True,
                        oracleType='known', PrCorrectRange=[0.25, 1.0]),
                Oracle(dataset, labels, classIndex, mapping, costVarying=True),
                Oracle(dataset, labels, classIndex, mapping, costVarying=True,
                        oracleType='known', CostRange=[0.01, 1.0])]

    for i in range(150):
        for oi, o in enumerate(oracles):
            y, c = o.query(i)
            print("Oracle %i: %s\t%.3f\t\t" % (oi + 1, str(y), c), end='')
        print()

    print("Done.")

