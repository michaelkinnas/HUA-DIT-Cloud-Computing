from pickle import load as pload
from typing import Callable, Optional
from torch.utils.data import Dataset
# from .im_tools import CIFAR_to_numpy
from cv2 import rotate, flip, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
from random import Random
from numpy import array
from os import walk
from PIL import Image


class CIFAR10Val(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, return_numpy=False, duplicate_ratio=0, transform_prob=0.0, random_seed=None):
        self.images = []        
        self.transform = transform
        self.return_numpy = return_numpy
        self.random = Random(random_seed)

        with open(f'{root}/cifar-10-batches-py/test_batch', 'rb') as fo:
            images_dict = pload(fo, encoding='bytes')
            for img, label in zip(images_dict[b'data'], images_dict[b'labels']):
                self.images.append((self.__CIFAR_to_numpy(img), array(label))) # shape HxWxC
                # self.labels.append(label)

        if duplicate_ratio != 0:
            if duplicate_ratio < 0:
                raise ValueError("Duplicate ratio must be greater than 0")

            # pick sample from images acording to ratio
            dup_images = []
            while duplicate_ratio > 1:
                dup_images += self.images
                duplicate_ratio -= 1
            self.images += self.random.sample(self.images, k = int(duplicate_ratio * len(self.images)))
            self.images += dup_images

            # shuffle images
            self.random.shuffle(self.images)

        #make random rotations and flips
        if transform_prob != 0:
            if transform_prob < 0 or transform_prob > 1:
                raise ValueError("Transform probability must be in the range (0, 1]")
                 
            for i, (image, label) in enumerate(self.images):
                if self.random.random() <= transform_prob:
                    image = self.random.choice([rotate(image, ROTATE_90_COUNTERCLOCKWISE), 
                                                rotate(image, ROTATE_90_CLOCKWISE), 
                                                rotate(image, ROTATE_180),
                                                flip(image, 0)])
                    
                    self.images[i] = (image, label)
    
    
    def __CIFAR_to_numpy(self, bin_data):
        r = array(bin_data[0:1024]).reshape((32, 32))
        g = array(bin_data[1024:2048]).reshape((32, 32))
        b = array(bin_data[2048:3072]).reshape((32, 32))   
        return array([r,g,b]).transpose((1, 2, 0))


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):      
        img, target = self.images[idx][0], self.images[idx][1]
        
        if self.transform:
            img = self.transform(img)
        
        if self.return_numpy:
            return img, target, self.images[idx][0]
       
        return img, target



# Use subset
class ImageNetVal(Dataset):
    def __init__(self, image_path: str, transform: Optional[Callable]=None, return_numpy=False, duplicate_ratio=0, transform_prob=0.0, random_seed=None):
        self.images = []        
        self.transform = transform
        self.return_numpy = return_numpy
        self.random = Random(random_seed)

        for i, (path, dirs, files) in enumerate(sorted(walk(image_path))):
            if len(files) == 0:
                continue

            for image in files:
                image = Image.open(f"{path+'/'+image}")
                if image.mode not in ['RGB']:
                    image = image.convert('RGB')
                self.images.append((array(image), array(i-1)))


        if duplicate_ratio != 0:
            if duplicate_ratio < 0:
                raise ValueError("Duplicate ratio must be greater than 0")
            
            # pick sample from images acording to ratio
            dup_images = []
            while duplicate_ratio > 1:
                dup_images += self.images
                duplicate_ratio -= 1

            # If transform prob make random rotations
            if transform_prob != 0:
                if transform_prob < 0 or transform_prob > 1:
                    raise ValueError("Transform probability must be in the range (0, 1]")
                    
                for image, label in dup_images:
                    if self.random.random() <= transform_prob:
                        image = self.random.choice([rotate(image, ROTATE_90_COUNTERCLOCKWISE), 
                                                    rotate(image, ROTATE_90_CLOCKWISE), 
                                                    rotate(image, ROTATE_180),
                                                    flip(image, 0)])
            
            # Duplicate remaining images (those that are less than 1)
            self.images += self.random.sample(self.images, k = int(duplicate_ratio * len(self.images)))            
            
            self.images += dup_images

        # shuffle images
        self.random.shuffle(self.images)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, target = self.images[idx][0], self.images[idx][1]

        if self.transform:
            img = self.transform(img)
        
        if self.return_numpy:
            return img, target, self.images[idx][0]
       
        return img, target