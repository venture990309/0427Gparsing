# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import importlib
import datetime
import random
import sys
import json
import glob
from visualization import generate_label

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-l", "--level",  type=int, required=False, default=None)
parser.add_argument("-n", "--model_name", type=str, required=True)
parser.add_argument("-m", "--mask", default=None, type=str)
parser.add_argument("-s", "--size", default=None, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 1

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  net = importlib.import_module('model.'+args.model_name)
  model = set_device(net.InpaintGenerator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['netG'])
  # model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)

  path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  for idx, (names, pose2, pose3, parsing1, parsing2, parsing3) in enumerate(dataloader):
    print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      gpu, idx, len(dataloader), names[0]))
    pose2, pose3, parsing1, parsing2, parsing3 = set_device([pose2, pose3, parsing1, parsing2, parsing3])

    with torch.no_grad():
      pred_img1 = model(pose2, parsing1)
      pred_img2 = model(pose3, parsing1)

    parsing1_vis = generate_label(parsing1, 256, 256)
    parsing2_vis = generate_label(parsing2, 256, 256)
    parsing3_vis = generate_label(parsing3, 256, 256)
    fake_parsing2_vis = generate_label(pred_img1, 256, 256)
    fake_parsing3_vis = generate_label(pred_img2, 256, 256)

    p1 = postprocess(parsing1_vis)
    p2 = postprocess(parsing2_vis)
    p3 = postprocess(parsing3_vis)
    fake_p2 = postprocess(fake_parsing2_vis)
    fake_p3 = postprocess(fake_parsing3_vis)
    for i in range(len(p1)):
      Image.fromarray(p1[i]).save(os.path.join(path, '{}_p1.png'.format(names[i].split('.')[0])))
      Image.fromarray(p2[i]).save(os.path.join(path, '{}_p2.png'.format(names[i].split('.')[0])))
      Image.fromarray(p3[i]).save(os.path.join(path, '{}_p3.png'.format(names[i].split('.')[0])))
      Image.fromarray(fake_p2[i]).save(os.path.join(path, '{}_fp2.png'.format(names[i].split('.')[0])))
      Image.fromarray(fake_p3[i]).save(os.path.join(path, '{}_fp3.png'.format(names[i].split('.')[0])))

  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  config['model_name'] = args.model_name
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 
