#!/bin/bash

cd mpc/code/
# Mask data.
./bin/UnitTestClient 0 ../par_ecg/demo.par.0.txt &
sleep 2
./bin/UnitTestClient 1 ../par_ecg/demo.par.1.txt &
sleep 2
./bin/UnitTestClient 2 ../par_ecg/demo.par.2.txt &
sleep 2
#./bin/unit_test 3 ../par_ecg/demo.par.3.txt ../../../data/ecg/text_demo_5500/ &
wait