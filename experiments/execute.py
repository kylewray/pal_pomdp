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


datasets = [
            {'prefix': "iris", 'Bmin': 2, 'Bmax': 3, 'Bstep': 2, 'Bc': 1.0,
                'filename': "../experiments/iris/iris.data", 'classIndex': 4,
                'trainSize': 50, 'testSize': 100,
                'classifier': "svm", 'numIterations': 25},
            {'prefix': 'adult', 'Bmin': 2, 'Bmax': 30, 'Bstep': 2, 'Bc': 5.0,
                'filename': "../experiments/adult/adult_converted.data", 'classIndex': 14,
                'trainSize': 50, 'testSize': 1000,
                'classifier': "svm", 'numIterations': 25},
            {'prefix': "spambase", 'Bmin': 2, 'Bmax': 30, 'Bstep': 2, 'Bc': 5.0,
                'filename': "../experiments/spambase/spambase.data", 'classIndex': 57,
                'trainSize': 50, 'testSize': 1000,
                'classifier': "svm", 'numIterations': 25}
            ]


def execute(scenario):
    for dataset in datasets:
        data = list()
        budgets = range(dataset["Bmin"], dataset["Bmax"] + dataset["Bstep"], dataset["Bstep"])

        for B in budgets:
            B = float(B)

            print("Executing Simulation for Budget %.2f..." % (B))

            sim = Simulation(scenario, B, dataset['Bc'], dataset["filename"], dataset["classIndex"],
                                dataset["trainSize"], dataset["testSize"],
                                dataset["classifier"], dataset["numIterations"])
            names, accuracies, costs = sim.execute()

            if len(data) == 0:
                data = [[name] for name in names]

            for j in range(len(names)):
                data[j] += ["%.4f" % (np.mean(accuracies[j])), "%.4f" % (np.std(accuracies[j])), "%.4f" % (np.mean(costs[j])), "%.4f" % (np.std(costs[j]))]

        with open(os.path.join(thisFilePath, "results", "_".join([dataset["prefix"], scenario])) + ".csv", "a") as f:
            f.write("Algorithm," + ",".join(list(map(str, budgets))) + ",\n")
            for algorithm in data:
                for d in algorithm:
                    f.write("%s," % (d))
                f.write("\n")

if __name__ == "__main__":
    scenario = None
    try:
        scenario = sys.argv[1]
    except Exception:
        print("Syntax:          python execute.py <scenario>")
        print("Scenarios:")
        print("  default        Compares a PAL POMDP and all three 'Scenario' PALs with all four oracles.")
        print("  original_1     Compares a POMDP, the 'Scenario #1', and baseline PALs with two oracles with cluster-based probabilities: Normal and Reluctant.")
        print("  original_2     Compares a POMDP, the 'Scenario #2', and baseline PALs with two oracles with cluster-based probabilities: Normal and Fallible.")
        print("  original_3     Compares a POMDP, the 'Scenario #3', and baseline PALs with two oracles with cluster-based probabilities: Normal and Cost Varying.")
        print("  original_all   Compares a PAL POMDP, the all 'Scenario' PALs, and a random PAL with all three oracles with cluster-based probabilities: Reluctant, Fallible, and Cost Varying.")
        print("  known_1        Compares a POMDP, the 'Scenario #1', and baseline PALs with two oracles with known probabilities: Normal and Reluctant.")
        print("  known_2        Compares a POMDP, the 'Scenario #2', and baseline PALs with two oracles with known probabilities: Normal and Fallible.")
        print("  known_3        Compares a POMDP, the 'Scenario #3', and baseline PALs with two oracles with known probabilities: Normal and Cost Varying.")
        print("  known_all      Compares a PAL POMDP, the all 'Scenario' PALs, and a random PAL with all three oracles with known probabilities: Reluctant, Fallible, and Cost Varying.")
        print("  expanded       Compares a PAL POMDP and all three 'Scenario' PALs with four oracles combining two and three types with known probabilities.")
        print("  insanity       Compares a PAL POMDP and all three 'Scenario' PALs with four oracles combining all three types simultaneously with known probabilities.")
        print("  baseline       Compares a PAL POMDP and baseline PALs with all four oracles.")

    if scenario is not None:
        execute(scenario)

