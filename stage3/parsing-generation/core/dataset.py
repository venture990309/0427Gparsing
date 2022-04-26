import random
import warnings
from random import shuffle
import os
import math
import numpy as np
from PIL import Image, ImageFilter
from glob import glob
import cv2
import pandas as pd
from torch.utils.data import IterableDataset
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageDraw
import json
from torch.nn import functional as FF
from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_args, debug=False, split='train', level=None):
        super(Dataset, self).__init__()
        self.split = split
        self.level = level
        self.w, self.h = data_args['w'], data_args['h']
        #    content_file_path = '/data/hj/Projects/new_task/Pose-Transfer/fashion_data/trainp'
        #    filelist = os.listdir(content_file_path)
        #    pairs_file = pd.read_csv('/data/hj/Projects/new_task/Pose-Transfer/fashion_data/fasion-resize-pairs-train.csv')
        #    size = len(pairs_file)
        #    pairs = []
        #    for i in range(size):
        #      if pairs_file.iloc[i]['from'] in filelist:
        #        pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
        #        pairs.append(pair)

        #        self.data_style = [os.path.join(data_args['style_file'], i)
        #                             for i in np.genfromtxt(os.path.join(data_args['list_root'], split + '.lst'), dtype=np.str,
        #                                                    encoding='utf-8')]
        self.data_style = [os.path.join(data_args['style_file'], i)
                           for i in os.listdir('/data/hj/Projects/new_task/Pose-Transfer/fashion_data/train2/')]

        # self.data_style = [os.path.join(data_args['style_file'], i)
        #                    for i in np.genfromtxt(os.path.join(data_args['list_root'], split + '.lst'),
        #                                           dtype=np.str, encoding='utf-8')]

        #        self.data_style.sort()
        # self.data_style.sort()

        self.point_num = 18
        self.height = 256
        self.width = 256
        self.radius = 4
        self.mean = (0, 0, 0)
        self.sl = 0.02
        self.sh = 0.02
        self.r1 = 0.25
        self.transform = transforms.Compose([transforms.ToTensor()
                                                , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # if split == 'train':
        #   self.data = self.data*data_args['extend']
        #   shuffle(self.data)
        # if debug:
        #   self.data = self.data[:100]

    def __len__(self):
        return len(self.data_style)

    #
    # def set_subset(self, start, end):
    #   self.mask = self.mask[start:end]
    #   self.data = self.data[start:end]

    def __getitem__(self, index):
        # try:
        item = self.load_item(index)

        # except:
        #   print('loading error: ' + self.data[index])
        #   item = self.load_item(0)
        return item

    def load_item(self, index):
        # load image
        # print('---------------------------------------------path', img_path)
        content_name = self.data_style[index].replace('jpg', 'png').replace('train2', 'trainp')
        style_name = self.data_style[index]
        pose_name = self.data_style[index].replace('train2', 'pose').replace('.jpg', '.json')
        parsing = Image.open(content_name)  # 读取parsing图片
        style_img = Image.open(style_name)

        # pose
        with open(pose_name) as f:
            pose_data = json.load(f)
        pose_maps = torch.zeros((self.point_num, self.height, self.width))
        im_pose = Image.new('RGB', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)

        for i in range(self.point_num):
            one_map = Image.new('RGB', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            if '%d' % i in pose_data:
                pointX = pose_data['%d' % i][0]
                pointY = pose_data['%d' % i][1]
                if pointX > 1 or pointY > 1:
                    draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                                  pointY + self.radius), 'white', 'white')
                    pose_draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                                       pointY + self.radius), 'white', 'white')
            one_map = self.transform(one_map)[0]
            pose_maps[i] = one_map

        parsing = np.array(parsing).astype(np.long)
        parsing = torch.from_numpy(parsing)
        # label
        label = torch.zeros(1, 20)
        parsing_20 = torch.zeros(20, 256, 256)
        parsing_shift = torch.zeros(20, 256, 256)
        for i in range(20):
            parsing_20[i] += (parsing == i).float()
            parsing_shift[i] += (parsing == i).float()
            if i in parsing:
                label[0][i] = 1
        label = label.view(label.size(0), label.size(1), 1, 1).expand(label.size(0), label.size(1), 256, 256)
        label = label[0]

        with open(pose_name) as f:
            pose_data = json.load(f)
        area = parsing_shift.size()[1] * parsing_shift.size()[2]  # 整体区域 img.size()=tensor.Size([1,256,256])  c x h x w
        target_area = random.uniform(self.sl, self.sh) * area  # 黑盒 随机要抠掉的大小 w x h
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)  # 随机要抠掉的长宽比 h / w
        h = int(round(math.sqrt(target_area * aspect_ratio)))  # 黑盒的h
        w = int(round(math.sqrt(target_area / aspect_ratio)))  # 黑盒的w
        if w > parsing_shift.size()[2]:
            w = 256
        elif h > parsing_shift.size()[1]:
            h = 256
        # 随机取遮挡的pose部位
        random_point_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

        posedata_key = list(pose_data.keys())
        random_res = []
        for i in list(random_point_list):
            if '%d' % i in pose_data:
                random_res += '%d' % i
        if len(random_res) >= 3:
            random_point = random.sample(random_res, 3)
        else:
            random_res = posedata_key
            random_point = random.sample(random_res, 3)
        y = [0, 0, 0]
        x = [0, 0, 0]
        xl = [0, 0, 0]
        xr = [0, 0, 0]
        yl = [0, 0, 0]
        yr = [0, 0, 0]
        try:
            for i in range(3):
                y[i] = pose_data['%d' % int(random_point[i])][1]  # w 对应pose坐标w
                x[i] = pose_data['%d' % int(random_point[i])][0]  # h 对应pose坐标h
                if x[i] > 1 or y[i] > 1:
                    xl[i] = int(x[i] - h / 2)
                    #                 print('xl[i]',xl[i])
                    #                 print('xl0',xl[0],i)
                    if xl[i] < 0:
                        xl[i] = 0

                    xr[i] = int(x[i] + h / 2)
                    if xr[i] < 0:
                        xr[i] = 0

                    yl[i] = int(y[i] - w / 2)
                    if yl[i] < 0:
                        yl[i] = 0

                    yr[i] = int(y[i] + w / 2)
                    if yr[i] < 0:
                        yr[i] = 0
            #                 x1 = random.randint(0, img.size()[1] - h)                     # 黑盒 随机坐标x1  h
            #                 y1 = random.randint(0, img.size()[2] - w)                     # 黑盒 随机坐标y1  W
            for i in range(20):  # 黑盒赋值
                parsing_shift[i, yl[0]:yr[0], xl[0]:xr[0]] = self.mean[0]
                parsing_shift[i, yl[1]:yr[1], xl[1]:xr[1]] = self.mean[0]
                parsing_shift[i, yl[2]:yr[2], xl[2]:xr[2]] = self.mean[0]
        except:
            print(pose_name)
        x_shift = random.uniform(-0.5, 0.5)
        y_shift = random.uniform(-0.5, 0.5)
        theta = torch.tensor([
            [1, 0, x_shift],
            [0, 1, y_shift]
        ], dtype=torch.float)
        grid = FF.affine_grid(theta.unsqueeze(0), parsing_shift.unsqueeze(0).size())
        affine_parsing = FF.grid_sample(parsing_shift.unsqueeze(0).float(), grid, )
        parsing_hat = affine_parsing[0]
        return label, pose_maps, parsing_hat, parsing_20

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(dataset=self, batch_size=batch_size, drop_last=True)
            for item in sample_loader:
                yield item