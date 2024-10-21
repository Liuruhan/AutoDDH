import re
import cv2
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import logging

import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, data_dir, cls_n, seg_n):
        self.data_dir = data_dir
        self.cls_n = cls_n
        self.seg_n = seg_n
        self.ids = [splitext(file)[0] for file in listdir(data_dir + 'img/')
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img):
        img_nd = np.array(img)

        if len(img_nd.shape) == 2 :
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        #if img_trans.max() == 255:
        img_trans = img_trans / 255

        if img_trans.shape[0] == 4:
            img_return = img_trans[0:3, :, :]
        else:
            img_return = img_trans
        return img_return

    @classmethod
    def resampler(cls, ids):
        np.random.shuffle(ids)
        class_len = [0, 0, 0, 0]
        new_ids = []
        new_idx = [[], [], [], []]

        #print('id:', ids)
        for i in range(len(ids)):
            for j in range(4):
                if int(ids[i][:1]) == j:
                    class_len[j] += 1
        print(class_len)
        # rank = [len(new_idx[0]), len(new_idx[1]), len(new_idx[2]), len(new_idx[3])].sort()
        rank_idx = [index for index, value in sorted(list(enumerate(class_len)), key=lambda x:x[1])]

        for i in range(len(ids)):
            for j in range(4):
                if int(ids[i][:1]) == rank_idx[j]:
                    new_idx[j].append(ids[i])
        #print(len(new_idx[0]), len(new_idx[1]), len(new_idx[2]), len(new_idx[3]))

        for i in range(len(new_idx[3])):
            if i < len(new_idx[0]):
                for j in range(4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[0]) and i < len(new_idx[1]):
                for j in range(1):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(1, 4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[1]) and i < len(new_idx[2]):
                for j in range(2):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(2, 4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[2]) and i < len(new_idx[3]):
                for j in range(3):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(3, 4):
                    new_ids.append(new_idx[j][i])

        print(len(new_ids))
        return new_ids
    
    def __getitem__(self, i):
        id = self.ids[i]
        img_file = self.data_dir + 'img/' + id + '.png'
        img = cv2.imread(img_file)
        img = self.preprocess(img)

        seg_file = self.data_dir + 'seg/' + id + '.png'
        seg = np.array(cv2.imread(seg_file, 0)).astype(int)
        # onehot_seg = np.eye(self.seg_n, dtype=seg.dtype)[seg.clip(0, self.seg_n - 1)]
        # onehot_seg = onehot_seg.transpose((2, 0, 1))

        label = [int(id[:re.search('_', id).span()[0]])]
        label_tensor = torch.Tensor(label)

        return {'img': torch.from_numpy(img).type(torch.FloatTensor),
                'seg': torch.from_numpy(seg).type(torch.LongTensor),
                'target': label_tensor.type(torch.LongTensor)}
