#!/bin/bash

cd mpc/code/

# Run protocol.
./bin/TrainSecureECG 0 ../par_ecg/demo.par.0.txt &
sleep 5
./bin/TrainSecureECG 1 ../par_ecg/demo.par.1.txt &
sleep 5
./bin/TrainSecureECG 2 ../par_ecg/demo.par.2.txt &
wait