#!/bin/bash

python simulation.py original_1 30.0 1.0 ../experiments/iris/iris.data 4 30 50 svm 1
python simulation.py original_2 30.0 1.0 ../experiments/iris/iris.data 4 30 50 svm 1
python simulation.py original_3 30.0 1.0 ../experiments/iris/iris.data 4 30 50 svm 1

python simulation.py asdf 30.0 1.0 ../experiments/iris/iris.data 4 30 50 svm 1

python simulation.py insanity 15.0 1.0 ../experiments/adult/adult_converted.data 14 40 1000 svm 1

