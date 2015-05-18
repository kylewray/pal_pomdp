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

import numpy as np

thisFilePath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(thisFilePath, "..", "src"))
from simulation import Simulation


thisFilePath = os.path.dirname(os.path.realpath(__file__))


datasets = [{"prefix": "iris", "Bmin": 10, "Bmax": 10, "Bstep": 2,
                "filename": "../experiments/iris/iris.data", "classIndex": 4,
                "trainSize": 30, "testSize": 100,
                "classifier": "svm", "numIterations": 1}]

scenarios = ["original_1", "original_2", "original_3", "baseline", "everything"]

for dataset in datasets:
    for scenario in scenarios:
        data = list()
        budgets = range(dataset["Bmin"], dataset["Bmax"] + dataset["Bstep"], dataset["Bstep"])

        for B in budgets:
            B = float(B)

            sim = Simulation(scenario, B, dataset["filename"], dataset["classIndex"],
                                dataset["trainSize"], dataset["testSize"],
                                dataset["classifier"], dataset["numIterations"])
            names, accuracies, costs = sim.execute()

            if len(data) == 0:
                data = [[name] for name in names]

            for j in range(len(names)):
                data[j] += ["%.4f" % (np.mean(accuracies[j])), "%.4f" % (np.std(accuracies[j])), "%.4f" % (np.mean(costs[j])), "%.4f" % (np.std(costs[j]))]

        with open(os.path.join(thisFilePath, sys.argv[1], "_".join([dataset["prefix"], scenario])) + ".csv", "a") as f:
            f.write("Algorithm," + ",".join(list(map(str, budgets))) + ",\n")
            for algorithm in data:
                for d in algorithm:
                    f.write("%s," % (d))
                f.write("\n")

