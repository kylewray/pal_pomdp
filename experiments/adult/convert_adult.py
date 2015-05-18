import numpy as np
import csv


def load_data(filename):
    data = list()

    f = open(filename, "r")
    dataset = csv.reader(f)

    for row in dataset:
        data = data + [row]

    return data


data = load_data("adult.data")
#print(data)

convertIndexes = [1, 3, 5, 6, 7, 8, 9, 13, 14]

features = list()
for k in range(len(data[0])):
    f = None

    if k in convertIndexes:
        f = set()
        for i in range(len(data)):
            f |= {data[i][k]}
        f = list(f)

    features += [f]

print(features)


with open("adult_converted.data", "w") as f:
    for i in range(len(data)):
        row = list()

        for k in range(len(data[i])):
            if features[k] == None:
                row += [str(int(data[i][k]))]
            else:
                row += [str(features[k].index(data[i][k]))]

        f.write(",".join(row))

        if i < len(data) - 1:
            f.write("\n")

