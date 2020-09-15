#!/bin/bash

cd mpc/code/
# Mask data.
./bin/ShareDataRC 0 ../par_ecg_rc/demo.par.0.txt &
sleep 2
./bin/ShareDataRC 1 ../par_ecg_rc/demo.par.1.txt &
sleep 2
./bin/ShareDataRC 2 ../par_ecg_rc/demo.par.2.txt &
sleep 2
./bin/ShareDataRC 3 ../par_ecg_rc/demo.par.3.txt ../../../data/ecg/text_demo_rc/ &
wait
