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

import numpy as np
import numpy.linalg as npla

from sklearn.linear_model import LogisticRegression

from pal import PAL


class PALOriginalScenario1(object):
    """ A proactive learner following the original paper's Scenario #1. """

    def __init__(self, dataset, numClasses, oracles, Bc):
        """ The constructor for the PAL Scenario #1 class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
                Bc          --  A list of budgets for how much we can spend for each oracle in initial clustering.
        """

        self.numOracles = len(oracles)
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        self.CT = self.pal.Ctotal
        self.Cround = 0.0
        self.Q = list()

        # The threshold for including data points in Nx as part of computing the uncertainty weighted density.
        self.tUWD = 1.0

    def __str__(self):
        """ Return the name of this object.

            Returns:
                The name of the object.
        """

        return "PAL (Original): Scenario #1"

    def create(self):
        """ Called before it iterates over the data points. """

        pass

    def select(self):
        """ Select a new data point to label. """

        ULSetMinusQ = [x for x in range(self.pal.get_num_unlabeled()) if x not in self.Q]

        # If we have run out of points to label, then select nothing.
        if self.pal.get_num_unlabeled() == 0 or len(ULSetMinusQ) == 0:
            return None, None

        hatU = self._estimate_utility()

        kStar = np.array([hatU[ULSetMinusQ, k].max() for k in range(self.numOracles) if self.pal.is_normal(k) or self.pal.is_reluctant(k)]).argmax()
        xStar = ULSetMinusQ[hatU[ULSetMinusQ, kStar].argmax()]

        self.Cround += self.pal.get_cost(xStar, kStar)
        self.Q = self.Q + [xStar]

        #print("kStar: %i\t xStar: %i\t ULSetMinusQ:%s" % (kStar, xStar, str(ULSetMinusQ)))
        #print("Q:", self.Q)

        return kStar, self.pal.map_index_for_query(xStar)

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

    def _estimate_utility(self):
        """ Estimate the utilities by returning the most up-to-date \hat{U}.

            Returns:
                The updated \hat{U}.
        """

        hatU = np.zeros((self.pal.get_num_unlabeled(), self.numOracles))

        # Create the initial training subset of the data, and the unlabeled dataset as the prediction dataset.
        Xinitial, yinitial = self.pal.get_labeled_dataset()
        Xpredict = self.pal.get_unlabeled_dataset()

        # Train using a SVM.
        c = LogisticRegression(C=1e5)
        #c = SVC(kernel='rbf', max_iter=1000, probability=True)
        c.fit(Xinitial, yinitial)
        PryGw = c.predict_proba(Xpredict)

        for x in range(self.pal.get_num_unlabeled()):
            PryGxw = PryGw[x, :]

            for k in range(self.numOracles):
                if self.pal.is_reluctant(k):
                    # This imposes the additional penalty for reluctant oracles, exactly as described in the original paper.
                    # Due to the strange update behavior here (it is a step behind), as well as in the original paper's algorithm pseudocode,
                    # we implemented it here as it was *intended* and *described* in writing in the original paper. It was likely
                    # just a logical oversight in the pseudocode definition, since otherwise it would have effectively divided by zero, or
                    # used the reluctant oracle's cost Ck twice (once as just Ck, then again as Cround which would have equaled Ck) until
                    # Cround finally rose to 2*Ck, 3*Ck, etc.
                    hatU[x, k] = self.pal.get_pr_answer(x, k) * self._uncertainty_weighted_density(x, PryGxw) / (self.pal.get_cost(x, k) + self.Cround)
                else:
                    hatU[x, k] = self.pal.get_pr_answer(x, k) * self._uncertainty_weighted_density(x, PryGxw) / self.pal.get_cost(x, k)

        return hatU

    def update(self, label, cost):
        """ Record the label and cost. """

        if label != None:
            # Note: self.Q[-1] = xStar, which is the selected data point index in the unordered dataset.
            self.pal.set_label(self.Q[-1], label)
            self.pal.update()

            self.CT += self.Cround
            self.Cround = 0.0
            self.Q = list()

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()

        self.CT += self.Cround
        self.Cround = 0.0
        self.Q = list()

        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.CT = self.pal.Ctotal
        self.Cround = 0.0
        self.Q = list()


class PALOriginalScenario2(object):
    """ A proactive learner following the original paper's Scenario #2. """

    def __init__(self, dataset, numClasses, oracles, Bc):
        """ The constructor for the PAL Scenario #2 class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
                Bc          --  A list of budgets for how much we can spend for each oracle in initial clustering.
        """

        self.numOracles = len(oracles)
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        self.CT = self.pal.Ctotal

        # The threshold for including data points in Nx as part of computing the uncertainty weighted density.
        self.tUWD = 1.0

    def __str__(self):
        """ Return the name of this object.

            Returns:
                The name of the object.
        """

        return "PAL (Original): Scenario #2"

    def create(self):
        """ Called before it iterates over the data points. """

        pass

    def select(self):
        """ Select a new data point to label. """

        # If we have run out of points to label, then select nothing.
        if self.pal.get_num_unlabeled() == 0:
            return None, None

        hatU = self._estimate_utility()

        kStar = np.array([hatU[:, k].max() for k in range(self.numOracles) if self.pal.is_normal(k) or self.pal.is_fallible(k)]).argmax()
        xStar = hatU[:, kStar].argmax()

        self.xStar = xStar

        #print("kStar: %i\t xStar: %i" % (kStar, xStar))

        return kStar, self.pal.map_index_for_query(xStar)

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

    def _estimate_utility(self):
        """ Estimate the utilities by returning the most up-to-date \hat{U}.

            Returns:
                The updated \hat{U}.
        """

        hatU = np.zeros((self.pal.get_num_unlabeled(), self.numOracles))

        # Create the initial training subset of the data, and the unlabeled dataset as the prediction dataset.
        Xinitial, yinitial = self.pal.get_labeled_dataset()
        Xpredict = self.pal.get_unlabeled_dataset()

        # Train using a SVM.
        c = LogisticRegression(C=1e5)
        #c = SVC(kernel='rbf', max_iter=1000, probability=True)
        c.fit(Xinitial, yinitial)
        PryGw = c.predict_proba(Xpredict)

        for x in range(self.pal.get_num_unlabeled()):
            PryGxw = PryGw[x, :]

            for k in range(self.numOracles):
                hatU[x, k] = self.pal.get_pr_correct(x, k) * self._uncertainty_weighted_density(x, PryGxw) / self.pal.get_cost(x, k)

        return hatU

    def update(self, label, cost):
        """ Record the label and cost. """

        if label != None:
            self.pal.set_label(self.xStar, label)
            self.pal.update()

        self.CT += cost

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()

        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.CT = self.pal.Ctotal


class PALOriginalScenario3(object):
    """ A proactive learner following the original paper's Scenario #3. """

    def __init__(self, dataset, numClasses, oracles, Bc):
        """ The constructor for the PAL Scenario #3 class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
                Bc          --  A list of budgets for how much we can spend for each oracle in initial clustering.
        """

        self.numOracles = len(oracles)
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        self.CT = self.pal.Ctotal

        # The threshold for including data points in Nx as part of computing the uncertainty weighted density.
        self.tUWD = 1.0

    def __str__(self):
        """ Return the name of this object.

            Returns:
                The name of the object.
        """

        return "PAL (Original): Scenario #3"

    def create(self):
        """ Called before it iterates over the data points. """

        pass

    def select(self):
        """ Select a new data point to label. """

        # If we have run out of points to label, then select nothing.
        if self.pal.get_num_unlabeled() == 0:
            return None, None

        hatU = self._estimate_utility()

        kStar = np.array([hatU[:, k].max() for k in range(self.numOracles) if self.pal.is_normal(k) or self.pal.is_cost_varying(k)]).argmax()
        xStar = hatU[:, kStar].argmax()

        self.xStar = xStar

        #print("kStar: %i\t xStar: %i" % (kStar, xStar))

        return kStar, self.pal.map_index_for_query(xStar)

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

    def _estimate_utility(self):
        """ Estimate the utilities by returning the most up-to-date \hat{U}.

            Returns:
                The updated \hat{U}.
        """

        hatU = np.zeros((self.pal.get_num_unlabeled(), self.numOracles))

        # Create the initial training subset of the data, and the unlabeled dataset as the prediction dataset.
        Xinitial, yinitial = self.pal.get_labeled_dataset()
        Xpredict = self.pal.get_unlabeled_dataset()

        # Train using a SVM.
        c = LogisticRegression(C=1e5)
        #c = SVC(kernel='rbf', max_iter=1000, probability=True)
        c.fit(Xinitial, yinitial)
        PryGw = c.predict_proba(Xpredict)

        for x in range(self.pal.get_num_unlabeled()):
            PryGxw = PryGw[x, :]

            for k in range(self.numOracles):
                hatU[x, k] = self._uncertainty_weighted_density(x, PryGxw) - self.pal.get_cost(x, k)

        return hatU

    def update(self, label, cost):
        """ Record the label and cost. """

        if label != None:
            self.pal.set_label(self.xStar, label)
            self.pal.update()

        self.CT += cost

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()

        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.CT = self.pal.Ctotal

