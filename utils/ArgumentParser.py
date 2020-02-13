# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ArgumentParser:
    def __init__(self,):
        pass
        
    def parse_args(self,):
        parser = argparse.ArgumentParser(description='Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # update !!
        parser.add_argument('--experimenter', dest='experimenter', help='experimenter', default='JSH', type=str)
        
        # gpu option
        parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu', default='0', type=str)
        parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', default=32, type=int)
        
        # model option
        parser.add_argument('--option', dest='option', default='b0', type=str)
        
        # train technology
        parser.add_argument('--weight_decay', dest='weight_decay', help='weight_decay', default=1e-4, type=float)
        parser.add_argument('--ema_decay', dest='ema_decay', help='ema', default=0.999, type=float)

        parser.add_argument('--log_iteration', dest='log_iteration', help='log_iteration', default=100, type=int)
        parser.add_argument('--val_iteration', dest='val_iteration', help='val_iteration', default=10000, type=int)
        parser.add_argument('--max_iteration', dest='max_iteration', help='max_iteration', default=100000, type=int)
        
        return vars(parser.parse_args())


