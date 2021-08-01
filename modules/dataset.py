import numpy as np
import os

import time
import glob
import math

from PIL import Image
import random 

from torch.utils.data import Dataset

import skimage
#from skimage.segmentation import find_boundaries
#import numpy as np
#import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
#from skimage.transform import rescale, resize, downscale_local_mean

EXTENSIONS = ['.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    flnm = "{bs}{ext}".format(bs = basename, ext = extension)
    return os.path.join(root, flnm)
    #return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

w0 = 5
sigma = 2

class NeuronEM2012(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, cvParam = 0.75, Train = True):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = []

        all_file = os.path.join(self.images_root, '*.png')

        filename = glob.glob(all_file)
        numel = len(filename)
        print('reacched here!')
        print(math.floor(numel*cvParam))

        if Train:
           for i in range(0, math.floor(numel*cvParam)):
               self.filenames.append(image_basename(filename[i]))
        else:
          for i in range(math.floor(numel*cvParam)+1, numel):
               self.filenames.append(image_basename(filename[i]))
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        separate = filename.split('_')
        #separate[1] = 'manual1'
        #conv = '_'.join(separate)
        with open(image_path(self.labels_root, separate[0], '_manual1.png'), 'rb') as f:
            label = load_image(f).convert('L')

        ######## Generate Pixel Weight Map ###########
        label_np = np.array(label)
        inv_label = skimage.util.invert(label_np)
        lbl_dst = distance_transform_edt(inv_label)         
        lbl_map = w0 * np.exp((-1 * (lbl_dst ** 2)) / (2 * (sigma ** 2)))
         
        label_map = Image.fromarray(lbl_map)

        # apply extra augmentation (same random operation on both image and label)
        
        # horizontal flip
        image = image.resize((512,512), resample=Image.BICUBIC)
        label = label.resize((512,512), resample=Image.NEAREST)
        label_map = label_map.resize((512,512), resample=Image.BICUBIC)

        flip_prob = random.random()
        if flip_prob<0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT) 
            label.transpose(Image.FLIP_LEFT_RIGHT)
            label_map.transpose(Image.FLIP_LEFT_RIGHT)
        # rotation:
        deg = random.randint(-90, 90)
        image = image.rotate(deg, resample=Image.BICUBIC)
        label = label.rotate(deg, resample=Image.NEAREST)
        label_map = label_map.rotate(deg, resample=Image.BICUBIC)
        

        if self.input_transform is not None:
            image = self.input_transform(image)
            label_map = self.input_transform(label_map)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, label_map
    def __len__(self):
        return len(self.filenames)


class Glands(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None, cvParam = 0.8, Train = True):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = []

        all_file = os.path.join(self.images_root, '*.png')

        filename = sorted(glob.glob(all_file))
        numel = len(filename)
        #print('hello!')
        #print(numel)
        #print(cvParam) 
        #print(math.floor(numel*cvParam))
        #print('bye!')
        if Train:
           for i in range(0,math.floor(numel*cvParam)):
               self.filenames.append(image_basename(filename[i]))
        else:
          for i in range(math.floor(numel*cvParam)+1, numel):
               self.filenames.append(image_basename(filename[i]))

        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '_1stHO.png'), 'rb') as f:
            label = load_image(f).convert('L')

        # apply extra augmentation (same random operation on both image and label)
        image = image.resize((960,960), resample=Image.BICUBIC)
        label = label.resize((960,960), resample=Image.NEAREST)
        
        # horizontal flip
        flip_prob = random.random()
        if flip_prob<0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT) 
            label.transpose(Image.FLIP_LEFT_RIGHT)
        # rotation:
        deg = random.randint(-90, 90)
        image = image.rotate(deg, resample=Image.BICUBIC)
        label = label.rotate(deg, resample=Image.NEAREST)
        

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
        
    def __len__(self):
        return len(self.filenames)

'''
class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)
'''
