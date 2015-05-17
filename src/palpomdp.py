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
import numpy.linalg as npla
import random as rnd
import itertools as it

from sklearn.svm import SVC

from oracle import Oracle
from pal import PAL


class PALPOMDP(MOPOMDP):
    """ A class which models a proactive learning agent. """

    def __init__(self, dataset, numClasses, oracles, Bc):
        """ The constructor of the PAL POMDP class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
                Bc          --  A list of budgets for how much we can spend for each oracle in initial clustering.
        """

        super().__init__()

        self.states = None
        self.actions = None
        self.observations = None

        # The PAL object which contains the dataset, labelings, probabilities, costs, and oracle information.
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        # The threshold for including data points in Nx as part of computing the uncertainty weighted density.
        self.tUWD = 1.0

        # Variables used for executing policies.
        self.Gamma = None
        self.pi = None

        self.currentDataPointIndex = 0
        self.previousAction = None
        self.b = None

    def _entropy(self, Pr):
        """ Compute the entropy of the probability given.

            Parameters:
                Pr  --  The probability, as a 1-d numpy array.

            Returns:
                The entropy of the probability.
        """

        # Handle the special case when Pr[y] == 0 (or is close). Since lim Pr(y) log(Pr(y)) -> 0,
        # we use this convention.
        def entropy_func(y):
            if Pr[y] < 0.0000000001:
                return 0.0
            else:
                return Pr[y] * np.log(Pr[y])

        return -sum([entropy_func(y) for y in range(Pr.shape[0])])

    def _uncertainty_weighted_density(self, dataPointIndex, PryGxw):
        """ Compute the uncertainty weighted density given the data point.

            Parameters:
                dataPointIndex  --  The data point index, from the unlabeled dataset.
                PryGxw          --  Pr(y | x, w) is the probability of each label for this data point.

            Returns:
                The uncertainty weighted density of the data point provided.
        """

        Nx = [i for i in range(self.pal.get_num_unlabeled()) if npla.norm(self.pal.get_unlabeled(i) - self.pal.get_unlabeled(dataPointIndex)) < self.tUWD]
        return sum([np.exp(-pow(npla.norm(self.pal.get_unlabeled(dataPointIndex) - self.pal.get_unlabeled(k)), 2)) * self._entropy(PryGxw) for k in Nx])

    def _compute_reward(self, s, a, dataPointIndex):
        """ Compute the reward based on the oracle cost, the budget, and some measure of the value of information.

            Parameters:
                s               --  The state index.
                a               --  The action index.
                dataPointIndex  --  The data point index within the unlabeled data set.

            Returns:
                The reward which combines a number of variables.
        """

        # First off, if we failed to label the data point last time, provide zero reward for choosing
        # an unreliable oracle again. This of course can be extended, or generalized, to any number of
        # "times you observed an oracle fail to respond." Since we are dealing with the original problem
        # formulation which has oracles that simply will *not respond ever* to a data point, we don't
        # want to get stuck picking that oracle on that data point forever. The original paper's model
        # is a bit impractical in this regard, and our model is a proof-of-concept, which can be easily
        # adapted in any manner to accommodate *anything*; it is a POMDP after all.
        if self.states[s][2] == False and self.pal.is_reluctant(a):
            return 0.0

        # Create the initial training subset of the data.
        Xinitial, yinitial = self.pal.get_labeled_dataset()

        # Create the training subset of the data. This is what we are building the PAL POMDP to operate over.
        # We need to use the data, without any labels of course, to determine the 'value of information' component
        # of the reward. This is based on the 'estimated uncertainty of the current learning function.'
        Xtrain = self.pal.get_unlabeled_dataset()

        # Train using a SVM.
        c = SVC(kernel='rbf', max_iter=1000, probability=True)
        c.fit(Xinitial, yinitial)
        PryGxw = c.predict_proba(Xtrain)[dataPointIndex, :]

        return self._uncertainty_weighted_density(dataPointIndex, PryGxw) / self.pal.get_cost(dataPointIndex, a)

    def create(self):
        """ Create the POMDP once the oracles and their probabilities have been defined. """

        # Create the states.
        self.states = list() #['Initial'] #, 'Success', 'Failure']
        for i in range(self.pal.get_num_unlabeled()):
            # For this data point (i.e., dataset[i, :]) create the corresponding states as a tuple:
            # < num correct, num incorrect, oracle responded >
            for numCorrect in range(i + 1):
                self.states += [(numCorrect, i - numCorrect, True), (numCorrect, i - numCorrect, False)]

        # The last set of states for the last data point only has true, which self-loops, since we are done.
        for numCorrect in range(self.pal.get_num_unlabeled() + 1):
            self.states += [(numCorrect, self.pal.get_num_unlabeled() - numCorrect, True)]

        self.n = len(self.states)

        # Create the actions.
        self.actions = [i for i in range(self.pal.get_num_oracles())]
        self.m = len(self.actions)

        # Create the state transitions.
        self.T = [[[0.0 for sp in range(self.n)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                for sp, statePrime in enumerate(self.states):
                    #print(state, action, statePrime)

                    if state[0] + state[1] == self.pal.get_num_unlabeled() and state == statePrime:
                        self.T[s][a][sp] = 1.0

                    elif state[0] == statePrime[0] and state[1] == statePrime[1] and \
                            ((state[2] == True and statePrime[2] == False) or \
                            (state[2] == False and statePrime[2] == False)):
                        # If no transition occurred and this was the first time the oracle did not respond,
                        # then we assign the probability of no answer (response) which equals
                        # 1 - Pr(answer | oracle, data point).
                        dataPointIndex = state[0] + state[1]
                        self.T[s][a][sp] = 1.0 - self.pal.get_pr_answer(dataPointIndex, action)

                    elif (state[0] + 1 == statePrime[0] and state[1] == statePrime[1]) and \
                            ((state[2] == True and statePrime[2] == True) or \
                            (state[2] == False and statePrime[2] == True)):
                        # If a successful labeling occurred, meaning that only the number correct increased,
                        # then we assign the probability of a correct labeling for this oracle, which equals
                        # Pr(answer | oracle, data point of s [not sp]) * Pr(correct | oracle, data point)
                        dataPointIndex = state[0] + state[1]
                        self.T[s][a][sp] = self.pal.get_pr_answer(dataPointIndex, action) * self.pal.get_pr_correct(dataPointIndex, action)

                    elif (state[0] == statePrime[0] and state[1] + 1 == statePrime[1]) and \
                            ((state[2] == True and statePrime[2] == True) or \
                            (state[2] == False and statePrime[2] == True)):
                        # If a failure of labeling occurred, meaning that only the number incorrect increased,
                        # then we assign the probability of an incorrect labeling for this oracle, which equals
                        # Pr(answer | oracle, data point of s [not sp]) * (1 - Pr(correct | oracle, data point))
                        dataPointIndex = state[0] + state[1]
                        self.T[s][a][sp] = self.pal.get_pr_answer(dataPointIndex, action) * (1.0 - self.pal.get_pr_correct(dataPointIndex, action))

                #print(sum([self.T[s][a][sp] for sp in range(len(states))]))
                    
        self.T = np.array(self.T)

        # Create the observations.
        self.observations = [True, False]
        self.z = len(self.observations)

        # Create the observation transitions.
        self.O = [[[0.0 for o in range(self.z)] for sp in range(self.n)] for a in range(self.m)]
        for a, action in enumerate(self.actions):
            for sp, statePrime in enumerate(self.states):
                for o, observation in enumerate(self.observations):
                    if (statePrime[2] == True and observation == True) or (statePrime[2] == False and observation == False):
                        self.O[a][sp][o] = 1.0

                #print(sum([self.O[a][sp][o] for o in range(len(observations))]))

        self.O = np.array(self.O)

        # Create the belief points.
        self.B = list()

        # Add the pure belief for each state.
        for s, state in enumerate(self.states):
            b = [0.0 for i in range(self.n)]
            b[s] = 1.0
            self.B += [b]

        # Add the mixed belief among each collection of states which are "probabilistically entangled."
        for i in range(self.pal.get_num_unlabeled()):
            numNonZeroBelief = float(i + 1)

            for answered in [True, False]:
                b = [0.0 for i in range(self.n)]
                for numCorrect in range(i + 1):
                    s = self.states.index((numCorrect, i - numCorrect, answered))
                    b[s] = 1.0 / numNonZeroBelief
                self.B += [b]

        # Important: We do *not* need to add belief points for the final set of states; these states have
        # no meaningful action to take there, since they are all absorbing with zero reward.

        self.r = len(self.B)
        self.B = np.array(self.B)

        # Create the reward functions.
        self.k = 1
        self.R = [[[0.0 for a in range(self.m)] for s in range(self.n)] for i in range(self.k)]
        for s, state in enumerate(self.states):
            # Note: You are currently at (s[0] + s[1] - 1), but since we pay an action to label the
            # *next* data point, we are really looking at paying the dataPointIndex + 1.
            dataPointIndex = (state[0] + state[1] - 1) + 1

            # If this was the last data point row (which are absorbing states), then no cost.
            if dataPointIndex == self.pal.get_num_unlabeled():
                continue

            for a, action in enumerate(self.actions):
                self.R[0][s][a] = self._compute_reward(s, a, dataPointIndex)

        self.R = np.array(self.R)

        self.horizon = self.pal.get_num_unlabeled() * 2 # Assume each can fail to respond once.

        self._compute_optimization_variables()

    def solve(self):
        """ Override the solve function to instead store the policy internally. """

        self.Gamma, self.pi = super().solve()

        # Initially, there is a collapsed belief over the first state.
        self.b = np.array([0.0 for s in self.states])
        self.b[0] = 1.0

        #print(self.Gamma)
        #print(self.pi)

    def _max_alpha_vector(self):
        """ Return the best value and action at the current internal belief point, using the internally stored policy. """

        v = np.matrix(self.Gamma) * np.matrix(self.b).T
        return np.max(v), self.pi[np.argmax(v)]

    def select(self):
        """ Based on the current belief, select an oracle to label the next belief.

            Returns:
                action                  --  The oracle index to label the current data point, whatever that may be.
                currentDataPointIndex   --  The index of the data point in the *Xtrain* dataset.
        """

        # If we have run out of points to label, then select nothing.
        if self.currentDataPointIndex == self.pal.get_num_unlabeled():
            return None, None

        val, action = self._max_alpha_vector()
        self.previousAction = action
        return action, self.pal.map_index_for_query(self.currentDataPointIndex)

    def update(self, label, cost):
        """ Update the belief with this new information.

            Parameters:
                label   --  The label observed (either 0, ..., num classes - 1, or None).
                cost    --  The cost associated with the previous request.
        """

        # Update the labeled and unlabeled datasets in the PAL object, as well as the cost.
        if label != None:
            self.pal.set_label(self.currentDataPointIndex, label)
            self.currentDataPointIndex += 1

        #print("Current Data Point Index: %i" % (self.currentDataPointIndex))

        # Define the observation.
        obs = None
        if label != None:
            obs = 0 # The observation was 'True', which is index 0.
        else:
            obs = 1 # The observation was 'False', which is index 1.

        # Update the belief point.
        bNew = np.array([self.O[self.previousAction, spIter, obs] * np.dot(self.T[:, self.previousAction, spIter], self.b) for spIter in range(self.n)])
        self.b = bNew / bNew.sum()

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()
        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.currentDataPointIndex = 0
        self.previousAction = None
        self.b = None

#    def simulate(self, Gamma, pi, outputHistory=False):
#        """ Simulate the execution of a given policy. Optionally output the action-observation history. This
#            stores the history for use in creating a final classifier.
#
#            Parameters:
#                Gamma           --  The alpha-vectors for the policy.
#                pi              --  The corresponding actions for each alpha-vector of the policy.
#                outputHistory   --  Optionally output the action-observation history. Default is False.
#
#            Returns:
#                The labels found while executing the policy.
#        """
#
#        # We know the true state of the system initially.
#        b = np.array([0.0 for s in range(self.n)])
#        b[0] = 1.0
#        s = 0
#
#        ylabels = [None for i in range(self.trainSize)]
#        dataPoint = 0
#
#        while dataPoint < self.trainSize:
#            # Take the action.
#            v, a = self._max_alpha_vector(b, Gamma, pi)
#
#            print("Action: %i" % (a))
#
#            # Stochastically transition the state.
#            sp = None
#            target = rnd.random()
#            current = 0.0
#            for spIter in range(self.n):
#                current += self.T[s, a, spIter]
#                if current >= target:
#                    sp = spIter
#                    break
#
#            # Obtain the label, modeling: non-responsive, success, and failure, respectively.
#            if self.states[s][0] == self.states[sp][0] and self.states[s][1] == self.states[sp][1]:
#                ylabels[dataPoint] = None
#            elif self.states[s][0] + 1 == self.states[sp][0] and self.states[s][1] == self.states[sp][1]:
#                ylabels[dataPoint] = self.dataset[self.trainIndexes[dataPoint], self.classIndex]
#                dataPoint += 1
#            elif self.states[s][0] == self.states[sp][0] and self.states[s][1] + 1 == self.states[sp][1]:
#                # Failed, so randomly pick one of the other classes.
#                correctClass = int(self.dataset[self.trainIndexes[dataPoint], self.classIndex])
#                wrongClasses = list(set(range(self.numClasses)) - {correctClass})
#                ylabels[dataPoint] = rnd.sample(wrongClasses, 1)[0]
#                dataPoint += 1
#
#            # Stochastically make an observation.
#            o = None
#            target = rnd.random()
#            current = 0.0
#            for oIter in range(self.z):
#                current += self.O[a, sp, oIter]
#                if current >= target:
#                    o = oIter
#                    break
#
#            # Update the belief point.
#            bNew = np.array([self.O[a, spIter, o] * np.dot(self.T[:, a, spIter], b) for spIter in range(self.n)])
#            b = bNew / bNew.sum()
#
#            # Transition the state!
#            s = sp
#
#        return np.array(ylabels)

#    def train(self, ytrain):
#        """ Train the final classifier after a simulation has generated a history of actions and observations.
#
#            Parameters:
#                ytrain  --  The proactively learned train labels.
#
#            Returns:
#                The accuracy of the model.
#        """
#
#        # Split apart the features and class labels.
#        nonClassIndexes = [i for i in range(self.dataset.shape[1]) if i != self.classIndex]
#        dataset = self.dataset[:, nonClassIndexes]
#        labels = self.dataset[:, self.classIndex]
#
#        # Create the training subset of the data.
#        Xtrain = dataset[self.trainIndexes, :]
#
#        # We do not have these anymore to train. We only have what was simulated as part of the PAL POMDP policy.
#        #ytrain = labels[self.trainIndexes]
#
#        # Train using k-nearest neighbors, logistic regression, or an SVM.
#        c = None
#        if self.classifier == 'knn':
#            c = KNeighborsClassifier(5)
#        elif self.classifier == 'logistic_regression':
#            c = LogisticRegression(C=1e5)
#        elif self.classifier == 'svm':
#            c = SVC(kernel='rbf', max_iter=1000)
#        else:
#            print("Error: Invalid classifier '%s'." % (self.classifier))
#            raise Exception()
#        c.fit(Xtrain, ytrain)
#
#        # Create the test set.
#        Xtest = dataset[self.testIndexes, :]
#        ytest = labels[self.testIndexes]
#
#        # Predict for each of these.
#        yprediction = c.predict(Xtest)
#
#        # Compute accuracy!
#        accuracy = np.sum(ytest == yprediction) / ytest.size
#
#        return accuracy
#
#    def truth(self):
#        """ Train a classifier using the *true* labels to compare the accuracy.
#
#            Returns:
#                The accuracy of the model.
#        """
#
#        labels = self.dataset[:, self.classIndex]
#        ytrain = labels[self.trainIndexes]
#        return self.train(ytrain)
#
#    def test(self):
#        """ Test the accuracy of the final classifier once it has been trained. """
#
#        pass


if __name__ == "__main__":
    print("Performing PALPOMDP Unit Test...")

    mapping = range(45)

    oracles = [Oracle("../experiments/iris/iris_small.data", 4, mapping),
                Oracle("../experiments/iris/iris_small.data", 4, mapping, reluctant=True),
                Oracle("../experiments/iris/iris_small.data", 4, mapping, fallible=True),
                Oracle("../experiments/iris/iris_small.data", 4, mapping, costVarying=True)]

    dataset = oracles[0].dataset.copy()

    Bc = [1.0, 1.0, 1.0, 1.0]

    palpomdp = PALPOMDP(dataset, 3, oracles, Bc)

    palpomdp.create()
    print(palpomdp)

    palpomdp.solve()
    print("Gamma:\n", palpomdp.Gamma)
    print("pi:\n", palpomdp.pi.tolist())

    print("Done.")

