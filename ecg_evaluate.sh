#!/bin/bash

python ./bin/ecg_evaluate_loss_wait_torch.py -m 62.0 -s 11.0 -c -b 32 -model cnnavg -cp same
# python ./bin/ecg_evaluate_loss_wait_torch.py -c -b 10 -li 10
# python ./bin/ecg_evaluate_loss_wait_torch.py -c -b 5 -li 5