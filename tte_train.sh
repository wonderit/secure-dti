#!/bin/bash

cd mpc/code/

# Run protocol.
./bin/TrainSecureTTE 0 ../par_ecg_tte/demo.par.0.txt &
sleep 5
./bin/TrainSecureTTE 1 ../par_ecg_tte/demo.par.1.txt &
sleep 5
./bin/TrainSecureTTE 2 ../par_ecg_tte/demo.par.2.txt &
wait
