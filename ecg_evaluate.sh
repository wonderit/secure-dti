#!/bin/bash

python ./bin/ecg_evaluate_loss_wait_torch.py -m 61.9 -s 10.4 -c -b 32 -model cnnavg -cp same
# python ./bin/ecg_evaluate_loss_wait_torch.py -m 61.9 -s 10.4 -c -b 32 -model cnnavg_concat -cp same