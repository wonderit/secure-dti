#!/bin/bash

cd mpc/code/
# Mask data.
./bin/ShareDataECG 0 ../par_ecg_small/demo.par.0.txt &
sleep 2
./bin/ShareDataECG 1 ../par_ecg_small/demo.par.1.txt &
sleep 2
./bin/ShareDataECG 2 ../par_ecg_small/demo.par.2.txt &
sleep 2
./bin/ShareDataECG 3 ../par_ecg_small/demo.par.3.txt ../../../data/ecg/text_demo_5500/ &
wait
