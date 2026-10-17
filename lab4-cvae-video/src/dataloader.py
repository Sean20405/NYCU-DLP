
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0, stride=None):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len
        self.stride = stride if stride is not None else video_len

        total_frames = int(len(self.img_folder) * self.partial)
        if total_frames < self.video_len:
            self.start_indices = []
        elif mode == 'train':
            # Sliding window for train mode to improve temporal coverage.
            self.start_indices = list(range(0, total_frames - self.video_len + 1, self.stride))
        else:
            # Keep non-overlapping chunks for validation.
            self.start_indices = list(range(0, total_frames - self.video_len + 1, self.video_len))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, index):
        start_idx = self.start_indices[index]
        
        imgs = []
        labels = []
        for i in range(self.video_len):
            frame_idx = start_idx + i
            label_list = self.img_folder[frame_idx].split('/')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[frame_idx]
            label_name = '/'.join(label_list)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)
    
    

