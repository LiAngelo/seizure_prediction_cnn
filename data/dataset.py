import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class Ictal(data.Dataset):

    def __init__(self, root, model_name, transforms=None, train=True, test=False):
        '''
        Get all images, split the data into train, test
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        '''
        self.test = test
        self.model_name = model_name
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test: data/test/
        # train: data/train/inter_7_0_1.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('_')[-3]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('_')[-3]))

        imgs_num = len(imgs)

        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            normalize_1d = T.Normalize(mean=[0.406],
                                       std=[0.225])

            if self.test or not train:
                if model_name == 'CNN_1d' or 'CNN_text':
                    self.transforms = T.Compose([
                        T.Grayscale(num_output_channels=1),
                        T.Scale((256, 6)),
                        T.RandomSizedCrop((224, 6)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize_1d
                    ])
                elif model_name == 'CNN_2d':
                    self.transforms = T.Compose([
                        T.Grayscale(num_output_channels=1),
                        T.Scale((256, 6)),
                        T.RandomSizedCrop((224, 6)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize
                    ])
            else:
                if model_name == 'CNN_1d' or 'CNN_text':
                    self.transforms = T.Compose([
                        T.Grayscale(num_output_channels=1),
                        T.Scale((256, 6)),
                        T.RandomSizedCrop((224, 6)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize_1d

                    ])
                elif model_name == 'CNN_2d':
                    self.transforms = T.Compose([
                        T.Grayscale(num_output_channels=1),
                        T.Scale((256, 6)),
                        T.RandomSizedCrop((224, 6)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize
                    ])

    def __getitem__(self, index):
        """
        return the info of each image
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('_')[-3])
        else:
            label = 1 if 'pre' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        if self.model_name != 'CNN_2d':
            data = np.squeeze(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
