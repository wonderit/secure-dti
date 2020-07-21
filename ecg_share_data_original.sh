#!/bin/bash

cd mpc/code/
# Mask data.
./bin/ShareDataECG 0 ../par_ecg_original/demo.par.0.txt &
sleep 2
./bin/ShareDataECG 1 ../par_ecg_original/demo.par.1.txt &
sleep 2
./bin/ShareDataECG 2 ../par_ecg_original/demo.par.2.txt &
sleep 2
./bin/ShareDataECG 3 ../par_ecg_original/demo.par.3.txt ../../../data/ecg/text_original_5500/ &
wait
