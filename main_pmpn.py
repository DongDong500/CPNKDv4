import os
import traceback
from datetime import datetime

from utils import MailSend
from kdTrain import train
from args import get_argparser, save_argparser
import utils

def exp(opts):
    params = save_argparser(opts, os.path.join(opts.default_prefix, opts.current_time))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpus

    start_time = datetime.now()

    mlog = {}
    slog = {}

    mlog['Short Memo'] = opts.short_memo + '\n'
    slog['Short Memo'] = opts.short_memo + '\n'

    mlog['Experiments'] = train(opts)
    slog['Experiments'] = {
        'Overall F1[0] mean/std' : mlog['Experiments']['Overall F1[0] mean/std'],
        'Overall F1[1] mean/std' : mlog['Experiments']['Overall F1[1] mean/std']
    }
    slog['dir'] = opts.current_time

    params['Overall F1[0] mean/std'] = mlog['Experiments']['Overall F1[0] mean/std']
    params['Overall F1[1] mean/std'] = mlog['Experiments']['Overall F1[1] mean/std']

    time_elapsed = datetime.now() - start_time

    mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
    slog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
    params["time_elpased"] = str(time_elapsed)
    
    utils.save_dict_to_json(d=mlog, json_path=os.path.join(opts.default_prefix, opts.current_time, 'mlog.json'))
    utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))

    # Transfer results by G-mail
    MailSend(subject = "Short report-%s" % "CPN segmentation exp results", 
                msg = slog,
                login_dir = opts.login_dir,
                ID = 'singkuserver',
                to_addr = ['sdimivy014@korea.ac.kr']).send()

if __name__ == '__main__':

    try:
        is_error = False
        is_abort = False
        
        short_memo = ['pmpn base study n=20 (non deterministic)']
        dataset = ['pmpn']
        mu = [122.5]
        std = [0.0]

        assert (len(short_memo) == len(mu) == len(std))

        total_time = datetime.now()
        for i in range(len(short_memo)):
            opts = get_argparser()
            opts.s_dataset = dataset[i]
            opts.exp_itr=20
            opts.short_memo = short_memo[i]
            opts.c_mu = mu[i]
            opts.c_std = std[i]
            exp(opts)
        
    except KeyboardInterrupt:
        is_abort = True
        print("Stop !!!")        
    except Exception as e:
        is_error = True
        print("Error", e)
        print(traceback.format_exc())

    if is_error:
        os.rename(os.path.join(opts.default_prefix, opts.current_time), os.path.join(opts.default_prefix, opts.current_time + '_error'))
    if is_abort:
        os.rename(os.path.join(opts.default_prefix, opts.current_time), os.path.join(opts.default_prefix, opts.current_time + '_aborted'))    
    
    total_time = datetime.now() - total_time
    print('Time elapsed (h:m:s.ms) {}'.format(total_time))