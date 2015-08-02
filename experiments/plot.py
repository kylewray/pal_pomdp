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
import csv

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


thisFilePath = os.path.dirname(os.path.realpath(__file__))


nameMap = {"PAL (POMDP): Value Divided by Cost": "PAL POMDP",
            "PAL (Original): Scenario #1": "PAL #1",
            "PAL (Original): Scenario #2": "PAL #2",
            "PAL (Original): Scenario #3": "PAL #3",
            "PAL (Baseline): Random": "Random",
            "PAL (Baseline): Fixed (Oracle 1)": "Fixed (o1)",
            "PAL (Baseline): Fixed (Oracle 2)": "Fixed (o2)",
            "PAL (Baseline): Fixed (Oracle 3)": "Fixed (o3)",
            "PAL (Baseline): Fixed (Oracle 4)": "Fixed (o4)"}

scenarioMap = {"original_1": "Original #1",
                "original_2": "Original #2",
                "original_3": "Original #3",
                "insanity": "Complex #1",
                "expanded": "Complex #2"}

colorMap = {"PAL (POMDP): Value Divided by Cost": "b",
            "PAL (Original): Scenario #1": "g",
            "PAL (Original): Scenario #2": "r",
            "PAL (Original): Scenario #3": "m",
            "PAL (Baseline): Random": "k",
            "PAL (Baseline): Fixed (Oracle 1)": "k",
            "PAL (Baseline): Fixed (Oracle 2)": "k"}

linestyleMap = {"PAL (POMDP): Value Divided by Cost": "-",
            "PAL (Original): Scenario #1": "--",
            "PAL (Original): Scenario #2": "--",
            "PAL (Original): Scenario #3": "--",
            "PAL (Baseline): Random": ":",
            "PAL (Baseline): Fixed (Oracle 1)": "-.",
            "PAL (Baseline): Fixed (Oracle 2)": "-."}

markerMap = {"PAL (POMDP): Value Divided by Cost": "o",
            "PAL (Original): Scenario #1": "^",
            "PAL (Original): Scenario #2": "<",
            "PAL (Original): Scenario #3": ">",
            "PAL (Baseline): Random": "s",
            "PAL (Baseline): Fixed (Oracle 1)": "+",
            "PAL (Baseline): Fixed (Oracle 2)": "x"}


def plot_data(directory, dataset, scenario, legendLocation='lower_right'):
    # Find all the files in this directory, which match the dataset-scenario pair.
    files = [f for f in os.listdir(directory) if f.split("_")[0] == dataset and \
                                                (f.split("_")[1] == scenario or \
                                                f.split("_")[1] + "_" + f.split("_")[2] == scenario)]

    if len(files) == 0:
        print("Error: Failed to find any files which matched dataset '%s' and scenario '%s'." % (dataset, scenario))
        raise Exception()

    # These variables will hold all information within the files.
    budgets = list()
    names = list()
    accuracies = dict()
    costs = dict()
    queries = dict()

    # Load each of the files' data into its correct locations.
    for filename in files:
        print("Loading file '%s'." % (filename))

        with open(os.path.join(thisFilePath, directory, filename), 'r') as f:
            data = csv.reader(f)

            current = None

            # First, load the header which contains the budgets.
            for i, row in enumerate(data):
                for j, element in enumerate(row):
                    # Skip blank cells.
                    if len(element) == 0:
                        continue

                    if i == 0 and j == 0:
                        pass
                    elif i == 0 and j > 0:
                        budgets += [int(element)]
                    elif j == 0:
                        if str(element) not in names:
                            names += [str(element)]
                        current = str(element)
                    elif (j - 1) % 6 == 0:
                        try:
                            accuracies[current] += [float(element)]
                        except KeyError:
                            accuracies[current] = [float(element)]
                    elif (j - 1) % 6 == 2:
                        try:
                            costs[current] += [float(element)]
                        except KeyError:
                            costs[current] = [float(element)]
                    elif (j - 1) % 6 == 4:
                        try:
                            queries[current] += [float(element)]
                        except KeyError:
                            queries[current] = [float(element)]

    # Sort each of them by the budget.
    order = np.argsort(budgets)
    budgets = np.array(budgets)[order]
    for name in names:
        accuracies[name] = np.array(accuracies[name])[order]
        costs[name] = np.array(costs[name])[order]
        queries[name] = np.array(queries[name])[order]

    # Determine the minimum y-value.
    ymin = int((min([accuracies[name].min() for name in names]) - 0.1) * 10.0) / 10.0
    ymax = int((max([accuracies[name].max() for name in names]) + 0.1) * 10.0) / 10.0

    # Set the font to be larger.
    font = {'family': 'normal', 'weight': 'bold', 'size': 22}
    mpl.rc('font', **font)
    plt.gcf().subplots_adjust(bottom=0.15)

    # With the loaded data, create plots!
    plt.title((dataset + " " + scenarioMap[scenario] + " Accuracy").title())
    plt.hold(True)

    plt.xlabel("Budget")
    plt.xticks(np.arange(budgets.min(), budgets.max() + 3.0, 3.0))
    plt.xlim([int(budgets.min() * 10.0) / 10.0, (int(budgets.max() * 10.0) + 1) / 10.0])

    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0.0, 1.0 + 0.1, 0.1))
    plt.ylim([ymin, ymax])
    plt.hlines(np.arange(0.0, 1.0 + 0.1, 0.1), budgets.min() - 1.0, budgets.max() + 1.0, colors=[(0.7, 0.7, 0.7)])

    for name in names:
        # Note: We match or improve upon these simple baselines, and they just make the already small
        # plots more confusing in the paper. You can plot them if you want though by commenting this
        # if statement.
        if name == "PAL (Baseline): Fixed (Oracle 1)" or name == "PAL (Baseline): Fixed (Oracle 2)":
            continue

        plt.plot(budgets, accuracies[name], label=nameMap[name],
                    linestyle=linestyleMap[name], linewidth=4,
                    marker=markerMap[name], markersize=14,
                    color=colorMap[name])
    if legendLocation == 'upper_left':
        plt.legend(loc=2)
    elif legendLocation == 'upper_right':
        plt.legend(loc=1)
    elif legendLocation == 'lower_left':
        plt.legend(loc=3)
    elif legendLocation == 'lower_right':
        plt.legend(loc=4)
    plt.show()

    #plt.title((dataset + " " + scenario + ": Cost").title())
    #plt.hold(True)
    #plt.xlim([budgets.min(), budgets.max()])
    #for name in names:
    #    plt.plot(budgets, costs[name], label=nameMap[name])
    #plt.legend(loc=4)
    #plt.show()

    #plt.title((dataset + " " + scenario + ": Query").title())
    #plt.xlim([budgets.min(), budgets.max()])
    #plt.hold(True)
    #for name in names:
    #    plt.plot(budgets, queries[name], label=nameMap[name])
    #plt.legend(loc=4)
    #plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 4:
        plot_data(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
    elif len(sys.argv) == 5:
        plot_data(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
    else:
        print("python plot.py <directory> <dataset> <scenario> <legend location, optional>")
        print("\tNote: 'legend location' in {upper_left, upper_right, lower_left, lower_right}")

