#!/bin/bash

python simulation.py original_1 30.0 ../experiments/iris/iris.data 4 30 50 svm 1
python simulation.py original_2 30.0 ../experiments/iris/iris.data 4 30 50 svm 1
python simulation.py original_3 30.0 ../experiments/iris/iris.data 4 30 50 svm 1
python simulation.py asdf 30.0 ../experiments/iris/iris.data 4 30 50 svm 1

