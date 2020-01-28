#!/bin/bash

cd mpc/code/

# Run protocol.
./bin/TrainSecureMNIST 0 ../par_mnist/demo.par.0.txt &
sleep 2
./bin/TrainSecureMNIST 1 ../par_mnist/demo.par.1.txt &
sleep 2
./bin/TrainSecureMNIST 2 ../par_mnist/demo.par.2.txt &
wait