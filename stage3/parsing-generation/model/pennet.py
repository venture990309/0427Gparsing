''' Pyramid-Context Encoder Networks: PEN-Net
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import use_spectral_norm
from core.partialconv2d import PartialConv2d
import functools
from core.tools import *
from torch.nn.utils.spectral_norm import spectral_norm



class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()

  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''

    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)

class InpaintGenerator(BaseNetwork):
  def __init__(self, init_weights=True):  # 1046
    super(InpaintGenerator, self).__init__()
    cnum = 32

    # attention module
    self.flow_param = {'input_dim': 58, 'dim': 64, 'n_res': 2, 'activ': 'relu',
                       'norm_conv': 'bn', 'norm_flow': 'none', 'pad_type': 'reflect', 'use_sn': False}
    self.f_gen = FlowGen(**self.flow_param)

    if init_weights:
      self.init_weights()


  def forward(self, pose,label, parsing_hat,):
    # encoder
    outputs = self.f_gen(torch.cat((pose, label, parsing_hat), dim=1))   # pose label affine parsing

    return outputs

class FlowGen(nn.Module):
  def __init__(self, input_dim=58, dim=64, n_res=2, activ='relu',
               norm_flow='ln', norm_conv='in', pad_type='reflect', use_sn=True):
    super(FlowGen, self).__init__()

    self.conv_column = ConvColumn(58, dim, n_res, activ,
                                  norm_conv, pad_type, use_sn)

  def forward(self, x):
    images_out = self.conv_column(x)  #    pose label affine parsing
    return images_out

class ConvColumn(nn.Module):
  def __init__(self, input_dim=58, dim=64, n_res=2, activ='lrelu',
               norm='ln', pad_type='reflect', use_sn=True):
    super(ConvColumn, self).__init__()

    self.down_sample = nn.ModuleList()
    self.up_sample = nn.ModuleList()

    self.down_sample += [nn.Sequential(
      Conv2dBlock(input_dim, dim, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
      )]

    self.down_sample += [nn.Sequential(
      Conv2dBlock(dim, dim * 2, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
      )]

    self.down_sample += [nn.Sequential(
      Conv2dBlock(2 * dim, 4 * dim, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
    )]

    self.down_sample += [nn.Sequential(
      Conv2dBlock(4 * dim, 8 * dim, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
    )]

    self.down_sample += [nn.Sequential(
      Conv2dBlock(8 * dim, 16 * dim, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
    )]

    self.down_sample += [nn.Sequential(
      Conv2dBlock(16 * dim, 16 * dim, 3, 2, 1, norm, activ, pad_type, use_sn=use_sn)
    )]

    dim = 16 * dim

    # content decoder
    self.up_sample += [(nn.Sequential(
      ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim, dim, 3, 1, 1, norm, activ, pad_type, use_sn=use_sn)))]  # 4-8

    self.up_sample += [(nn.Sequential(
      nn.Upsample(scale_factor=2),
      Conv2dBlock(2 * dim, dim // 2, 3, 1, 1, norm, activ, pad_type, use_sn=use_sn)))]  # 8-16

    self.up_sample += [(nn.Sequential(
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim, dim // 4, 3, 1, 1, norm, activ, pad_type, use_sn=use_sn)))] # 16-32

    self.up_sample += [(nn.Sequential(
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim // 2, dim // 8, 3, 1, 1, norm, activ, pad_type, use_sn=use_sn)))] # 32-64

    self.up_sample += [(nn.Sequential(
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim // 4, dim // 16, 3, 1, 1, norm, activ, pad_type, use_sn=use_sn)))] # 64-128

    self.up_sample += [(nn.Sequential(
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim // 8, 20, 3, 1, 1, norm, 'none', pad_type, use_sn=use_sn)))]  # 128-256

    self.sogtmax = nn.Softmax(dim=1)

  def forward(self, x):


    x1 = self.down_sample[0](x)  # torch.Size([4, 128, 128, 128])
    x2 = self.down_sample[1](x1)  # 64
    x3 = self.down_sample[2](x2) # torch.Size([4, 512, 32, 32])
    x4 = self.down_sample[3](x3) # 16
    x5 = self.down_sample[4](x4) # 8
    x6 = self.down_sample[5](x5) # 4


    up5 = self.up_sample[0](x6)
    u5 = torch.cat((up5, x5), 1)
    up4 = self.up_sample[1](u5)
    u4 = torch.cat((up4, x4), 1)
    up3 = self.up_sample[2](u4)
    u3 = torch.cat((up3, x3), 1)
    up2 = self.up_sample[3](u3)
    u2 = torch.cat((up2, x2), 1)
    up1 = self.up_sample[4](u2)
    u1 = torch.cat((up1, x1), 1)
    up0 = self.up_sample[5](u1)
    images_out = self.sogtmax(up0)
    return images_out


class ResBlocks(nn.Module):
  def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
    super(ResBlocks, self).__init__()
    self.model = []
    for i in range(num_blocks):
      self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
    self.model = nn.Sequential(*self.model)

  def forward(self, x):
    return self.model(x)

class ResBlock(nn.Module):
  def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
    super(ResBlock, self).__init__()

    model = []
    model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
    model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class Conv2dBlock(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size, stride,
               padding=0, norm='none', activation='relu', pad_type='zero', dilation=1,
               use_bias=True, use_sn=False):
    super(Conv2dBlock, self).__init__()
    self.use_bias = use_bias
    # initialize padding
    if pad_type == 'reflect':
      self.pad = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
      self.pad = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
      self.pad = nn.ZeroPad2d(padding)
    else:
      assert 0, "Unsupported padding type: {}".format(pad_type)

    # initialize normalization
    norm_dim = output_dim
    if norm == 'bn':
      self.norm = nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
      self.norm = nn.InstanceNorm2d(norm_dim)
    elif norm == 'ln':
      self.norm = LayerNorm(norm_dim)
    elif norm == 'adain':
      self.norm = AdaptiveInstanceNorm2d(norm_dim)
    elif norm == 'none':
      self.norm = None
    else:
      assert 0, "Unsupported normalization: {}".format(norm)

    # initialize activation
    if activation == 'relu':
      self.activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
      self.activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
      self.activation = nn.PReLU()
    elif activation == 'selu':
      self.activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
      self.activation = nn.Tanh()
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    elif activation == 'none':
      self.activation = None
    else:
      assert 0, "Unsupported activation: {}".format(activation)

    # initialize convolution
    if use_sn:
      self.conv = spectral_norm(
        nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
    else:
      self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

  def forward(self, x):
    x = self.conv(self.pad(x))
    if self.norm:
      x = self.norm(x)
    if self.activation:
      x = self.activation(x)
    return x

class AdaptiveInstanceNorm2d(nn.Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super(AdaptiveInstanceNorm2d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    # weight and bias are dynamically assigned
    self.weight = None
    self.bias = None
    # just dummy buffers, not used
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))

  def forward(self, x):
    assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
    b, c = x.size(0), x.size(1)
    running_mean = self.running_mean.repeat(b)
    running_var = self.running_var.repeat(b)

    # Apply instance norm
    x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

    out = F.batch_norm(
      x_reshaped, running_mean, running_var, self.weight, self.bias,
      True, self.momentum, self.eps)

    return out.view(b, c, *x.size()[2:])

  def __repr__(self):
    return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine

    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class Get_image(nn.Module):
  def __init__(self, input_dim, output_dim, activation='tanh'):
    super(Get_image, self).__init__()
    self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1,
                            padding=1, pad_type='reflect', activation=activation)

  def forward(self, x):
    return self.conv(x)


if __name__ == '__main__':
  import sys
