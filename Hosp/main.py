from epiweeks import Week
from data import get_epiweeks_list
from hosp import Hosp
import argparse
import os
import numpy as np
import traceback
from copy import copy
import os
import gc
import torch

# CUDA_VISIBLE_DEVICE=3 python main.py --region all --dev cuda --exp 400 --start_ew 202036 --end_ew 202109 --step 1


def train_predict(args):
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    if not os.path.exists('./results'):
        os.mkdir('./results') 
    
    model = Hosp(args)
    model.train_predict()

    model = None
    del model
    gc.collect()
    if args.dev != 'cpu':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--start_ew',type=str, default='202101',help='Start prediction week in CDC format')
    parser.add_argument('--end_ew',type=str, default='202101',help='End prediction week in CDC format')
    parser.add_argument('--region', nargs='+',default='all',help='Use all or a list of regions')
    parser.add_argument('--dev',type=str, default='cpu',help='')
    parser.add_argument('--exp',type=str, default='1',help='Experiment number/id')
    parser.add_argument('--step',type=str, default='1',help='Step between prediction weeks')
    parser.set_defaults(p=False)
    
    args = parser.parse_args()

    # get list of epiweeks for iteration
    start_ew = Week.fromstring(args.start_ew)
    end_ew = Week.fromstring(args.end_ew)
    iter_weeks = get_epiweeks_list(start_ew,end_ew)
    step = int(args.step)
    # handle prediction step
    if step != 1:
        id = np.arange(0,len(iter_weeks),step)
        iter_weeks = [iter_weeks[i] for i in id]

    region_list = args.region
    if len(region_list)==1:
        region_list = region_list[0]

    def run_all_weeks(args,region):
        args.region = region  # updating to conform correct input
        for ew in iter_weeks:
            args.pred_week = ew.cdcformat()
            try:
                train_predict(args) 
            except Exception as e:
                print(f'exception: did not work for {region} week {ew}: '+ str(e) + '\n')
                traceback.print_exc()
    
    run_all_weeks(copy(args),region_list)