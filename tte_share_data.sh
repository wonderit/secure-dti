#!/bin/bash

cd mpc/code/
# Mask data.
./bin/ShareDataTTE 0 ../par_ecg_tte/demo.par.0.txt &
sleep 2
./bin/ShareDataTTE 1 ../par_ecg_tte/demo.par.1.txt &
sleep 2
./bin/ShareDataTTE 2 ../par_ecg_tte/demo.par.2.txt &
sleep 2
./bin/ShareDataTTE 3 ../par_ecg_tte/demo.par.3.txt ../../../data/ecg/text_demo_tte/ &
wait
