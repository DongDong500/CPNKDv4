import os
import torch

import numpy as np
import random

from _train import _train


'''
    Return format

    mlog['Experiments'] = {
            "1" : {
                "F1-0" : 0.1, "F1-1" : 0.9
            },
            "2" : {
                "F1-0" : 0.1, "F1-1" : 0.9
            },
            "Overall F1[0] mean/std" : "0.2/0.01",
            "Overall F1[1] mean/std" : "0.4/0.01"
        }
'''

def train(opts) -> dict:

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s:" % (devices, opts.gpus))

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    test_result = {}
    for exp_itr in range(0, opts.exp_itr):        
        run_id = exp_itr
        test_result[run_id] = _train(opts, devices, run_id)
            
    f10 = 0
    f11 = 0
    s10 = 0
    s11 = 0
    N = 0
    for k in test_result.keys():
        N += 1
        f10 += test_result[k]['F1 [0]']
        f11 += test_result[k]['F1 [1]']
        s10 += test_result[k]['F1 [0]'] ** 2
        s11 += test_result[k]['F1 [1]'] ** 2
    f10 /= N
    f11 /= N
    s10 /= N
    s11 /= N
    # sample standard deviation
    s10 = np.sqrt((N/(N-1))*(s10 - f10**2))
    s11 = np.sqrt((N/(N-1))*(s11 - f11**2))

    test_result["Overall F1[0] mean/std"] = f"{f10}/{s10}"
    test_result["Overall F1[1] mean/std"] = f"{f11}/{s11}"
    
    return test_result