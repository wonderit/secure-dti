#!/bin/bash

cd mpc/code/
#
## Run protocol.
./bin/TrainSecureDTI 0 ../par/demo.par.0.txt &
sleep 1
./bin/TrainSecureDTI 1 ../par/demo.par.1.txt &
sleep 1
./bin/TrainSecureDTI 2 ../par/demo.par.2.txt &
wait
# Mask data.
#./bin/ShareData 0 ../par/demo.par.0.txt &
#./bin/ShareData 1 ../par/demo.par.1.txt &
#./bin/ShareData 2 ../par/demo.par.2.txt &
#./bin/ShareData 3 ../par/demo.par.3.txt ../../demo_data/batch_pw/ &
#wait
