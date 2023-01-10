import argparse
import yaml
import sys
import os
import torch
import numpy as np
from runners import *
import cv2

import os

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

arg_dict = {}
arg_dict['config'] = "kitti_bev.yml"
arg_dict['exp'] = 'debug'
arg_dict['ni'] = True

arg_dict['seed'] = 1234
arg_dict['comment'] = ''
arg_dict['doc'] = 'kitti'
arg_dict['verbose'] = False
arg_dict['test'] = False
arg_dict['nvs'] = False
arg_dict['fast_fid'] = False
arg_dict['resume_training'] = False
arg_dict['image_folder'] = 'images'

args = dict2namespace(arg_dict)
args.log_path = os.path.join(args.exp, 'logs', args.doc)
print(args.exp)
# parse config file
with open(os.path.join('configs', args.config), 'r') as f:
    config = yaml.safe_load(f)

new_config = dict2namespace(config)

# add device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# logging.info("Using device: {}".format(device))
new_config.device = device

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

from datasets.kitti import KITTI_BEV
os.environ['KITTI360_DATASET'] = '/sharedata/home/shared/jiangq/KITTI-360'
# '/sharedata/home/jiangq/DATA/kitti360_bev'
dataset = KITTI_BEV(preprocess_path=' ', normalize=True,config=new_config,split='train')
print(len(dataset))
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

# for i, data in enumerate(dataloader):
#     data = data[0].squeeze().numpy().transpose(1,2,0) * 255
#     print(data.max(axis=(0,1)),data.min(axis=(0,1)))
#     print(os.path.exists("image_results"))
#     print(cv2.imwrite("image_results/{}.jpg".format(i), data))
#     if i > 10: break


for i, (data,_) in enumerate(dataloader):
    data = data.squeeze().numpy()
    # np.save("/sharedata/home/jiangq/DATA/kitti360_bev/test/bev_test_{}".format(i), data)
    print(data.min(axis=(1,2)), data.max(axis=(1,2)))
    img_array = data.transpose(1,2,0) * 255
    print(cv2.imwrite("/sharedata/home/jiangq/DATA/kitti360_bev/sample_imgs/{}.jpg".format(i), img_array))
    if i > 3: break
