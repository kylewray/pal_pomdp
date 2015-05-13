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
from sklearn.svm import SVC

from sklearn.cluster import KMeans


NUM_ORACLES = 4
BASE_ORACLE_COST = 1.0 / 3.0


def oracle_values_random(labeledDataset, dataPoint):
    """ Return a string of the oracle values for the data point given.

        Parameters:
            labeledDataset  --  The small subset of the dataset which we have initially labeled. (Labels)
            dataPoint       --  One of the data points which we have *not* yet labeled. (No Label)

        Returns:
            The string of all the oracles' values.
    """

    oracleValues = ""

    # All Scenarios: The normal ("awesome") oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (rnd.uniform(0.8, 0.9), rnd.uniform(0.8, 0.9), BASE_ORACLE_COST)

    # Scenario #1: The reluctant oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (rnd.uniform(0.9, 1.0), rnd.uniform(0.6, 0.8), BASE_ORACLE_COST / 3.0)

    # Scenario #2: The imprecise oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (rnd.uniform(0.7, 0.8), rnd.uniform(0.9, 1.0), BASE_ORACLE_COST / 3.0)

    # Scenario #3: The varying (data point sensitive) oracle.
    oracleValues += "%.2f,%.2f,%.2f" % (rnd.uniform(0.8, 0.9), rnd.uniform(0.8, 0.9), rnd.uniform(BASE_ORACLE_COST / 3.0, BASE_ORACLE_COST * 3.0))

    return oracleValues


def oracle_values_cluster(dataset, labels, dataPointClusterDistance, maxClusterDistance, classifier):
    """ Return a string of the oracle values for the data point given.

        Parameters:
            dataset     --  The features of the small subset of the dataset which we have initially labeled.
            labels      --  The labels of the small subset of the dataset which we have initially labeled.
            numClasses  --  The number of classes in the dataset; used to set how many clusters there are.
            dataPoint   --  One of the data points which we have *not* yet labeled. (No label given here; just features.)
            classifier  --  The type of classifier to use for computing initial probabilities.

        Returns:
            The string of all the oracles' values.
    """

    COST_RATIO = 1.0 / 3.0

    PrAnswer = None

    PrCorrect = None

    CostNonUniform = None

    oracleValues = ""

    # All Scenarios: The normal ("awesome") oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (1.0, 1.0, BASE_ORACLE_COST)

    # Scenario #1: The reluctant oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (1.0, PrAnswer, BASE_ORACLE_COST * COST_RATIO))

    # Scenario #2: The imprecise oracle.
    oracleValues += "%.2f,%.2f,%.2f," % (PrCorrect, 1.0, BASE_ORACLE_COST * COST_RATIO))

    # Scenario #3: The varying (data point sensitive) oracle.
    oracleValues += "%.2f,%.2f,%.2f" % (1.0, 1.0, CostNonUniform))

    return oracleValues


def gen_pal(inputDataFile, outputPALFile, classIndex, classifier, initialSize, trainSize, testSize, kNCC):
    """ Generate a PAL file given the parameter specifications.

        Parameters:
            inputDataFile   --  The input data file and path.
            outputPALFile   --  The output data file and path.
            classIndex      --  The column index in the data file of the class variable.
            classifier      --  The classifier to use, e.g., knn.
            initialSize     --  The number of data points initially given with labels to the proactive learner. 
            trainSize       --  The number of data points to use for training.
            testSize        --  The number of data points to use for testing.
            kNCC            --  The k-value for normalized cross-correlation.
    """

    # Load the dataset.
    dataset = None
    try:
        dataset = np.genfromtxt(inputDataFile, delimiter=',')
    except Exception:
        print("Error: Failed to load data file '%s'." % (inputDataFile))
        return

    # Ensure valid input values.
    numDataPoints = np.shape(dataset)[0]
    if numDataPoints < trainSize + testSize + initialSize:
        print("Error: Number of data points is less than: initial size + train size + test size.")
        return

    if classIndex < 0 or classIndex >= np.shape(dataset)[1]:
        print("Error: Class index %i is not in [0, %i]." % (classIndex, np.shape(dataset)[1]))
        return

    # Determine a set of data points for initial, train, and test.
    indexes = list(range(numDataPoints))
    rnd.shuffle(indexes)

    initial = indexes[0:initialSize]
    train = indexes[initialSize:(initialSize + trainSize)]
    test = indexes[(initialSize + trainSize):(initialSize + trainSize + testSize)]

    # Useful variable which is all feature indexes that are not the class index.
    nonClassIndexes = [i for i in range(dataset.shape[1]) if i != classIndex]

    # Write the output file.
    with open(outputPALFile, 'w') as f:
        # First, write the header lines.
        f.write(",".join(map(str, [NUM_ORACLES, inputDataFile, classIndex, classifier, initialSize, trainSize, testSize, kNCC])) + "\n")
        f.write(",".join(map(str, initial)) + "\n")
        f.write(",".join(map(str, test)))

        # Method #2: Use a cluster method for the oracle value probability estimates.
        #cluster = KMeans(k=numClasses)
        #cluster.fit(dataset[initial, nonClassIndexes])

        # Next, for each data point, define the probabilities and costs associated with each oracle.
        for dataPointIndex in train:
            dataPoint = dataset[dataPointIndex, nonClassIndexes]

            # Method #1: Use reasonable random values.
            f.write("\n%i,%s" % (dataPointIndex, oracle_values_random(dataset[initial, :], dataPoint)))

            # Method #2: Use clustering to estimate reasonable probabilities.
            #f.write("\n%i,%s" % (dataPointIndex, oracle_values_cluster(dataset[initial, :], dataPoint)))


if __name__ == "__main__":
    try:
        gen_pal(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                sys.argv[4], int(sys.argv[5]), int(sys.argv[6]),
                int(sys.argv[7]), int(sys.argv[8]))
    except Exception:
        print("Syntax:   python gen_pal.py <input data file> <output PAL file> <class column index> <classifier: {knn, ???}> <initial size> <train size> <test size> <NCC k value>")


