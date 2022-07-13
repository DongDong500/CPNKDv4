import os
import torch

import numpy as np
import random

from _train import _train

def train(opts) -> dict:

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (devices, str(opts.gpus)))

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

    return {
        'F1 [0]' : f'{f10:.4f}, std: {s10:.4f}', 'F1 [1]' : f'{f11:.4f}, std: {s11:.4f}'
    }