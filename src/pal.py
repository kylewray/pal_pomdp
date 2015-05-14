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

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from oracle import Oracle


def sigmoid(x):
    L = 1.0
    k = 6.0
    x0 = 0.0
    return L / (1.0 + np.exp(-k * (x - x0)))


class PAL(object):
    """ A Proactive Learner (PAL) class to handle initialization of oracle probabilities and costs. """

    def __init__(self, dataset, oracles, Bc):
        """ The constructor for the PAL object, setting up variables given the dataset and oracles.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                oracles     --  A list of the Oracle objects to use.
                Bc          --  A list of budgets for how much we can spend for each oracle in initial clustering.
        """

        self.dataset = dataset.copy()
        self.numDataPoints = self.dataset.shape[0]
        self.numOracles = len(oracles)

        # Our goal will be to fill this variable with labels.
        self.labels = np.array([None for i in range(self.numDataPoints)])

        # The set of labeled and unlabeled data point indexes.
        self.L = list()
        self.UL = list(range(self.numDataPoints))

        # Pr(correct | data point, oracle) as 2-d array.
        self.PrCorrect = np.zeros((dataset.shape[0], self.numOracles))

        # Pr(answer | data point, oracle) as 2-d array.
        self.PrAnswer = np.zeros((dataset.shape[0], self.numOracles))

        # C(data point, oracle) as 2-d array.
        self.Cost = np.zeros((dataset.shape[0], self.numOracles))

        # Setup the default values.
        for j, o in enumerate(oracles):
            if o.reluctant:
                self.PrAnswer[:, j] = 0.5
            else:
                self.PrAnswer[:, j] = 1.0

            if o.fallible:
                self.PrCorrect[:, j] = 0.5
            else:
                self.PrCorrect[:, j] = 1.0

        # For each of the oracles, initialize the above variables.
        Ctotal = 0.0
        for j, o in enumerate(oracles):
            Ctotal += self._init_oracle(j, o, Bc[j])
        self.Ctotal = Ctotal

    def _init_oracle(self, oracleIndex, oracle, Bc):
        """ Initialize the probabilities and costs for the oracle provided.

            Parameters:
                oracleIndex --  The oracle's index.
                oracle      --  The oracle to initialize.
                Bc          --  The budget for this oracle.
        """

        # It is free to query for the costs, so just ask the oracle how much each data point is.
        for i in range(self.numDataPoints):
            self.Cost[i, oracleIndex] = oracle.query_cost(i)

        # Compute the average cost for this oracle.
        Cavg = self.Cost[:, oracleIndex].sum() / float(self.numDataPoints)

        # Given our budget, compute how many data point we will label.
        p = int(Bc / Cavg)

        # If budget is given, then we can only guess a prior of 0.5 or 1.0 probability
        # based on the type of oracle.
        if p == 0:
            return 0.0

        # Cluster the dataset into p clusters.
        kmeans = KMeans(n_clusters=p)
        kmeans.fit(self.dataset)

        # Find the distances from each data point to each cluster, then sort to know which
        # data points are closest to any cluster centroid. Also, note the data point index.
        clusterDistances = kmeans.transform(self.dataset[self.UL, :])
        pts = [[index, clusterDistances[i, :].argmin(), clusterDistances[i, :].min()] for i, index in enumerate(self.UL)]
        pts = [[[pt[0], pt[2]] for pt in pts if pt[1] == c] for c in range(p)]
        pts = [np.array(sorted(cpts, key=lambda x: x[1])) for cpts in pts]

        # Query for the top data points, as stored in the 'pts' variable.
        C = 0.0
        for cpts in pts:
            # Determine the data point index and distance to more easily read here.
            dataPointIndex = cpts[0][0]
            distance = cpts[0][1]

            # If the cost would put us above the budget, terminate.
            if C + oracle.query_cost(dataPointIndex) > Bc:
                break

            # Query the oracle to obtain that delicious beautiful data.
            self.labels[dataPointIndex], cost = oracle.query(dataPointIndex)

            C += cost

            #print(self.labels[dataPointIndex])

        # Update the set of labeled and unlabeled data points.
        self.L = list(set(self.L) | {cpts[0][0] for cpts in pts if self.labels[cpts[0][0]] != None})
        self.UL = list(set(self.UL) - set(self.L))

        #print(self.L)
        #print(self.UL)

        # For use in the computation of PrCorrect, we must build a classifier using the data points we have
        # labeled so far with this oracle.
        c = LogisticRegression(C=1e5)
        c.fit(self.dataset[self.L, :], self.labels[self.L])
        PrLogReg = c.predict_proba(self.dataset)

        # After our queries, we can iterate over the entire dataset and compute the probabilities.
        # We do this by iterating over each cluster, then each point within that cluster.
        for cpts in pts:
            # Determine the data point index and distance to more easily read here.
            clusterDataPointIndex = cpts[0][0]
            minDistance = cpts[0][1]
            maxDistance = cpts[-1][1]

            # For each point in this cluster.
            for pt in cpts:
                dataPointIndex = pt[0]
                distance = pt[1]

                # First we will compute PrAnswer. This requires defining h, which flips
                # the sign based on the return of a label or not. This only is set if
                # we have a reluctant oracle.
                if oracle.reluctant:
                    h = 1.0
                    if self.labels[clusterDataPointIndex] == None:
                        h = -1.0

                    self.PrAnswer[dataPointIndex, oracleIndex] = sigmoid(h * (1.0 - \
                                        (distance - minDistance) / (maxDistance - minDistance)))
                    #print(self.PrAnswer[dataPointIndex, oracleIndex])
                else:
                    self.PrAnswer[dataPointIndex, oracleIndex] = 1.0

                # Next we will compute PrCorrect. This requires defining h, which specifies
                # a value on [-1, 1] proportional to the max_y PrLogReg[y] using the logistic regression
                # result above. This only is set if we have a fallible oracle.
                if oracle.fallible:
                    h = 2.0 * PrLogReg[dataPointIndex, :].max() - 1.0

                    self.PrCorrect[dataPointIndex, oracleIndex] = sigmoid(h * (1.0 - \
                                        (distance - minDistance) / (maxDistance - minDistance)))
                else:
                    self.PrCorrect[dataPointIndex, oracleIndex] = 1.0

                #print("Data Point %i\t PrAnswer = %.3f\t PrCorrect = %.3f" % (dataPointIndex,
                #                        self.PrAnswer[dataPointIndex, oracleIndex], self.PrCorrect[dataPointIndex, oracleIndex]))
            #print("------------------")

        return C

    def labeled_dataset(self):
        """ Get the labeled dataset and its labels.

            Returns:
                The labeled dataset and the corresponding labels, as 2-d and 1-d numpy arrays, respectively.
        """

        return self.dataset[self.L, :], self.labels[self.L]

    def unlabeled_dataset(self):
        """ Get the unlabeled dataset.

            Returns:
                The unlabeled dataset as a 2-d numpy array.
        """

        return self.dataset[self.UL, :]


if __name__ == "__main__":
    print("Performing PAL Unit Test...")

    Ctotal = 0.0
    paymentCallback = lambda c: c

    oracles = [Oracle("../experiments/iris/iris.data", 4, paymentCallback),
                Oracle("../experiments/iris/iris.data", 4, paymentCallback, reluctant=True),
                Oracle("../experiments/iris/iris.data", 4, paymentCallback, fallible=True),
                Oracle("../experiments/iris/iris.data", 4, paymentCallback, variedCosts=True)]

    dataset = oracles[0].dataset.copy()

    Bc = [1.0, 1.0, 1.0, 1.0]

    pal = PAL(dataset, oracles, Bc)

    print("Oracle PrAnswer:\n", pal.PrAnswer)
    print("Oracle PrCorrect:\n", pal.PrCorrect)
    print("Oracle Cost:\n", pal.Cost)
    print("Clustering Cost / Clustering Budget: %.4f / %.4f" % (pal.Ctotal, sum(Bc)))

    print("Done.")

