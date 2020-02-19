#!/bin/bash

cd mpc/code/

# Run protocol.
./bin/TrainSecureECG 0 ../par_ecg_new/demo.par.0.txt &
sleep 2
./bin/TrainSecureECG 1 ../par_ecg_new/demo.par.1.txt &
sleep 2
./bin/TrainSecureECG 2 ../par_ecg_new/demo.par.2.txt &
wait