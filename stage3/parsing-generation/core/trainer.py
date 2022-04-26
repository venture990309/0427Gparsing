import os
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import Dataset
from core.utils import set_seed, set_device, Progbar, postprocess
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19, PerceptualCorrectness, TVLoss, cross_entropy
from core import metric as module_metric

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_images
from visualization import generate_label
from . import networks

tensorboard_dir = '/data/hj/Projects/new_task/xswz/parsing-generation/release_model/Gparsing/'
display_count = 20
board = SummaryWriter(tensorboard_dir)

from tensorboardX import SummaryWriter


class Trainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0
    if debug:
      self.config['trainer']['save_freq'] = 5
      self.config['trainer']['valid_freq'] = 5

    # setup data set and data loader
    self.train_dataset = Dataset(config['data_loader'])
    worker_init_fn = partial(set_seed, base=config['seed'])
    self.train_sampler = None
    if config['distributed']:
      self.train_sampler = DistributedSampler(self.train_dataset,
                                              num_replicas=config['world_size'], rank=config['global_rank'])
    self.train_loader = DataLoader(self.train_dataset,
                                   batch_size=config['trainer']['batch_size'] // config['world_size'],
                                   shuffle=(self.train_sampler is None), num_workers=config['trainer']['num_workers'],
                                   pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)

    # set up losses and metrics
    # self.adversarial_loss = set_device(AdversarialLoss(type=self.config['losses']['gan_type']))
    self.criterionGAN = set_device(networks.GANLoss(use_lsgan=True))
    self.dis_writer = None
    self.gen_writer = None
    self.num_D = 2
    self.summary = {}
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
      self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.train_args = self.config['trainer']

    net = importlib.import_module('model.' + config['model_name'])
    self.netG = set_device(net.InpaintGenerator())
    # self.netD = set_device(net.Discriminator(in_channels=6, use_sigmoid=config['losses']['gan_type'] != 'hinge'))
    # self.netD2 = set_device(net.Discriminator(in_channels=24, use_sigmoid=config['losses']['gan_type'] != 'hinge'))
    self.netD = set_device(networks.define_D(input_nc=20, ndf=64,
                                             netD='basic', n_layers_D=3,
                                             norm='instance', use_sigmoid=False, init_type='normal',
                                             init_gain=0.02, num_D=1, getIntermFeat=False,
                                             gpu_ids=[0]))
    self.optimG = torch.optim.Adam(self.netG.parameters(), lr=config['trainer']['lr'],
                                   betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self.optimD = torch.optim.Adam(self.netD.parameters(), lr=config['trainer']['lr'] * config['trainer']['d2glr'],
                                   betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self.load()
    if config['distributed']:
      self.netG = DDP(self.netG, device_ids=[config['global_rank']], output_device=config['global_rank'],
                      broadcast_buffers=True, find_unused_parameters=False)
      self.netD = DDP(self.netD, device_ids=[config['global_rank']], output_device=config['global_rank'],
                      broadcast_buffers=True, find_unused_parameters=False)

  # get current learning rate
  def get_lr(self, type='G'):
    if type == 'G':
      return self.optimG.param_groups[0]['lr']
    return self.optimD.param_groups[0]['lr']

  # learning rate scheduler, step
  def adjust_learning_rate(self):
    decay = 0.1 ** (min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
    new_lr = self.config['trainer']['lr'] * decay
    if new_lr != self.get_lr():
      for param_group in self.optimG.param_groups:
        param_group['lr'] = new_lr
      for param_group in self.optimD.param_groups:
        param_group['lr'] = new_lr

  # load netG and netD
  def load(self):
    model_path = self.config['save_dir']
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
      latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
      ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
      ckpts.sort()
      latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
    if latest_epoch is not None:
      gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
      dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
      opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
      if self.config['global_rank'] == 0:
        print('Loading model from {}...'.format(gen_path))
      data = torch.load(gen_path, map_location=lambda storage, loc: set_device(storage))
      self.netG.load_state_dict(data['netG'])
      data = torch.load(dis_path, map_location=lambda storage, loc: set_device(storage))
      self.netD.load_state_dict(data['netD'])
      data = torch.load(opt_path, map_location=lambda storage, loc: set_device(storage))
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')

  # save parameters every eval_epoch
  def save(self, it):
    if self.config['global_rank'] == 0:
      gen_path = os.path.join(self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
      dis_path = os.path.join(self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
      opt_path = os.path.join(self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
      print('\nsaving model to {} ...'.format(gen_path))
      if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
        netG, netD = self.netG.module, self.netD.module
      else:
        netG, netD = self.netG, self.netD
      torch.save({'netG': netG.state_dict()}, gen_path)
      torch.save({'netD': netD.state_dict()}, dis_path)
      torch.save({'epoch': self.epoch,
                  'iteration': self.iteration,
                  'optimG': self.optimG.state_dict(),
                  'optimD': self.optimD.state_dict()}, opt_path)
      os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

  def add_summary(self, writer, name, val):
    if name not in self.summary:
      self.summary[name] = 0
    self.summary[name] += val
    if writer is not None and self.iteration % 100 == 0:
      writer.add_scalar(name, self.summary[name] / 100, self.iteration)
      self.summary[name] = 0

  def cross_entropy(self, fake_out, real_parsing):
    # 取log
    fake_out = fake_out + 0.00001
    log_output = torch.log(fake_out)
    loss = - torch.sum((real_parsing * log_output))
    loss = loss / (256 * 256)
    return loss

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    mae = 0
    i = 0
    for label, pose_maps,parsing_hat,parsing_20 in self.train_loader:
      self.iteration += 1
      i = i + 1
      self.adjust_learning_rate()
      end = time.time()
      label, pose_maps, parsing_hat, parsing_20 = set_device([label, pose_maps,parsing_hat,parsing_20])

      fake_parsing = self.netG(label, pose_maps, parsing_hat)


      self.add_summary(self.dis_writer, 'lr/dis_lr', self.get_lr(type='D'))
      self.add_summary(self.gen_writer, 'lr/gen_lr', self.get_lr(type='G'))

      dis_loss = 0
      # image discriminator loss
      dis_real_feat = self.netD(parsing_20)
      # dis_real_featp = self.netD(img2 * mask_pp)
      dis_fake_feat = self.netD(fake_parsing.detach())
      # dis_fake_featp = self.netD(pred_img.detach() * mask_pp)
      dis_real_loss = self.criterionGAN(dis_real_feat, True)
      # dis_real_lossp = self.criterionGAN(dis_real_featp, True)
      dis_fake_loss = self.criterionGAN(dis_fake_feat, False)
      # dis_fake_lossp = self.criterionGAN(dis_fake_featp, False)
      dis = (dis_real_loss + dis_fake_loss) / 2
      dis_loss += dis
      self.add_summary(self.dis_writer, 'loss/dis_loss', dis.item())
      self.optimD.zero_grad()
      dis_loss.backward()
      self.optimD.step()

      # generator adversarial loss
      gen_fake_feat = self.netD(fake_parsing)
      gen_fake_loss = self.criterionGAN(gen_fake_feat, True)
      gen_loss = gen_fake_loss * self.config['losses']['adversarial_weight']
      gen_loss += gen_fake_loss
      self.add_summary(self.gen_writer, 'loss/gen_fake_loss', gen_loss.item())

      # generator l1 loss
      gen_loss = 0
      hole_loss = self.cross_entropy(fake_parsing, parsing_20)
      hole_loss = hole_loss * self.config['losses']['hole_weight']
      gen_loss += hole_loss
      self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

      # generator backward
      self.optimG.zero_grad()
      gen_loss.backward()
      self.optimG.step()
      
      fake_parsing_vis = generate_label(fake_parsing, 256, 256)
      parsing_hat_vis = generate_label(parsing_hat, 256, 256)
      parsing_20_vis = generate_label(parsing_20, 256, 256)  # batchx 3 x 256 x256
      visuals = [parsing_hat_vis, parsing_20_vis, fake_parsing_vis]

      if (self.iteration + 1) % display_count == 0:
        board_add_images(board, 'combine', visuals, i)
      board.close()

      # logs
      new_mae = (torch.mean(torch.abs(parsing_20 - fake_parsing))).item()
      mae = new_mae if mae == 0 else (new_mae + mae) / 2
      speed = parsing_20.size(0) / (time.time() - end) * self.config['world_size']
      logs = [("epoch", self.epoch), ("iter", self.iteration), ("lr", self.get_lr()),
              ('mae', mae), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(parsing_20) * self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

      # saving and evaluating
      if self.iteration % self.train_args['save_freq'] == 0:
        self.save(int(self.iteration // self.train_args['save_freq']))
      if self.iteration % self.train_args['valid_freq'] == 0:
        self._test_epoch(int(self.iteration // self.train_args['save_freq']))
        if self.config['global_rank'] == 0:
          print('[**] Training till {} in Rank {}\n'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.config['global_rank']))
      if self.iteration > self.config['trainer']['iterations']:
        break

  def _test_epoch(self, it):
    if self.config['global_rank'] == 0:
      print('[**] Testing in backend ...')
      model_path = self.config['save_dir']
      result_path = '{}/results_{}_level_03'.format(model_path, str(it).zfill(5))
      log_path = os.path.join(model_path, 'valid.log')
      try:
        os.popen('python test.py -c {} -n {} -l 3 -m {} -s {} > valid.log;'
                 'CUDA_VISIBLE_DEVICES=1 python eval.py -r {} >> {};'
                 'rm -rf {}'.format(self.config['config'], self.config['model_name'],
                                    self.config['data_loader']['mask'], self.config['data_loader']['w'],
                                    result_path, log_path, result_path))
      except (BrokenPipeError, IOError):
        pass

  def train(self):
    while True:
      self.epoch += 1
      if self.config['distributed']:
        self.train_sampler.set_epoch(self.epoch)
      self._train_epoch()
      if self.iteration > self.config['trainer']['iterations']:
        break
    print('\nEnd training....')
