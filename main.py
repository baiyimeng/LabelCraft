import argparse
from utils.Context import ContextManager, DatasetManager
import config.const as const
import model.Inputs as data_in
import torch
import numpy as np
import random 
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(542674)

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='./workspace')
parser.add_argument('--dataset_name', type=str, choices=['kuaishou, wechat'], default='kuaishou')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--model', type=str, help='which model to use', default='DIN')
parser.add_argument('--batch_size', type=int, help='training batch_size', default=4096)
parser.add_argument('--extra_name', type=str, help='extra model name', default='')
parser.add_argument('--adapt', type=float, choices=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], default=0.5)
parser.add_argument('--sample_ratio', type=float, choices=[0.5, 0.02], default=0.5)
parser.add_argument('--lr_rec', type=float, choices=[1e-1, 1e-2, 1e-3], default=1e-1)
parser.add_argument('--lr_label', type=float, default=1e-4)
parser.add_argument('--normalization', type=int, choices=[0,1], default=1)
parser.add_argument('--disable', type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

if args.dataset_name == 'kuaishou':
    const.init_dataset_setting_kuaishou()
    data_in.init_data_attribute_ls('kuaishou')
elif args.dataset_name == 'wechat':
    const.init_dataset_setting_wechat()
    data_in.init_data_attribute_ls('wechat')
else:
    raise ValueError(f'Not support dataset: {args.dataset_name}')

cm = ContextManager(args)
dm = DatasetManager(args)
trainer = cm.set_trainer(args, cm, dm)
trainer.train_and_test()