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

from pal import PAL


class PALBaselineRandom(object):
    """ A PAL baseline class that continually selects a random oracle at each time step. """

    def __init__(self, dataset, numClasses, oracles):
        """ The constructor for the PAL random (oracle selector) baseline class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
        """

        # This baseline doesn't waste resources with an initial clustering; it loves
        # gambling... a lot... and it wants all its resources for that.
        Bc = np.array([0.0 for o in oracles])

        self.numOracles = len(oracles)
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        self.currentDataPointIndex = 0

    def __str__(self):
        """ Return the name of this object.

            Returns:
                The name of the object.
        """

        return "PAL (Baseline): Random"

    def create(self):
        """ Called before it iterates over the data points. """

        pass

    def select(self):
        """ I get to pick a lucky random oracle! """

        # If we have run out of points to label, then select nothing.
        if self.currentDataPointIndex == self.pal.get_num_unlabeled():
            return None, None

        return rnd.randint(0, self.numOracles - 1), self.pal.map_index_for_query(self.currentDataPointIndex)

    def update(self, label, cost):
        """ Record the label and cost. """

        if label != None:
            self.pal.set_label(self.currentDataPointIndex, label)
            self.currentDataPointIndex += 1

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()
        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.currentDataPointIndex = 0


class PALBaselineFixed(object):
    """ A PAL baseline class that continually selects the same oracle at each time step. """

    def __init__(self, dataset, numClasses, oracles, fixedOracle):
        """ The constructor for the PAL fixed (oracle selector) baseline class.

            Parameters:
                dataset     --  The complete dataset without labels. (No labels are included; must query oracles.)
                numClasses  --  The number of classes.
                oracles     --  A list of the Oracle objects to use.
                fixedOracle --  The fixed oracle that this baseline keeps selecting.
        """

        # This baseline doesn't waste resources with an initial clustering; it loves
        # only one oracle, and it is fully committed to it.
        Bc = np.array([0.0 for o in oracles])

        self.myFavoriteOracle = fixedOracle
        self.pal = PAL(dataset, numClasses, oracles, Bc)

        self.currentDataPointIndex = 0

    def __str__(self):
        """ Return the name of this object.

            Returns:
                The name of the object.
        """

        return "PAL (Baseline): Fixed (Oracle %i)" % (self.myFavoriteOracle + 1)

    def create(self):
        """ Called before it iterates over the data points. """

        pass

    def select(self):
        """ I get to pick my favorite! """

        # If we have run out of points to label, then select nothing.
        if self.currentDataPointIndex == self.pal.get_num_unlabeled():
            return None, None

        return self.myFavoriteOracle, self.pal.map_index_for_query(self.currentDataPointIndex)

    def update(self, label, cost):
        """ Record the label and cost. """

        if label != None:
            self.pal.set_label(self.currentDataPointIndex, label)

        self.currentDataPointIndex += 1

    def finish(self):
        """ Perform any final finishing computation once the budget is spent or data points have all been labeled. Then, return the labeled dataset. """

        self.pal.update()
        return self.pal.get_labeled_dataset()

    def reset(self):
        """ Reset the variables controlling iteration. This does not reset the policy.  """

        self.currentDataPointIndex = 0

