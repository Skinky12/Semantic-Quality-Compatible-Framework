import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv


class SMRM_Folder(data.Dataset):

    def __init__(self, root, index, transform, istrain, dist_type=0, frname='none'):

        if istrain:
            img_names = scipy.io.loadmat(os.path.join(root, 'fnames_train.mat'))['fnames_train']
            img_scores = scipy.io.loadmat(os.path.join(root, '%s_sctr%02d.mat' % (frname, dist_type)))['sctr']
            sub_folder_name = 'train%02d' % dist_type
        else:
            img_names = scipy.io.loadmat(os.path.join(root, 'fnames_test.mat'))['fnames_test']
            img_scores = scipy.io.loadmat(os.path.join(root, '%s_scte%02d.mat' % (frname, dist_type)))['scte']
            sub_folder_name = 'test%02d' % dist_type

        sample = []

        for i, item in enumerate(index):
            sample.append((os.path.join(root, sub_folder_name, img_names[item, 0][0]), img_scores[item, 0]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)
        labels = np.array(mos_all).astype(np.float32)
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, '1024x768', imgname[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
