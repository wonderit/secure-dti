#!/bin/bash

cd mpc/code/

# Run protocol.
./bin/TrainSecureRC 0 ../par_ecg_rc/demo.par.0.txt &
sleep 5
./bin/TrainSecureRC 1 ../par_ecg_rc/demo.par.1.txt &
sleep 5
./bin/TrainSecureRC 2 ../par_ecg_rc/demo.par.2.txt &
wait
