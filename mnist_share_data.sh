#!/bin/bash

cd mpc/code/
# Mask data.
./bin/ShareDataMNIST 0 ../par_mnist/demo.par.0.txt &
sleep 2
./bin/ShareDataMNIST 1 ../par_mnist/demo.par.1.txt &
sleep 2
./bin/ShareDataMNIST 2 ../par_mnist/demo.par.2.txt &
sleep 2
./bin/ShareDataMNIST 3 ../par_mnist/demo.par.3.txt ../../../data/MNIST/text_demo_1280/ &
wait