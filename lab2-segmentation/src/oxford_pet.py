from glob import glob
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import albumentations as A


class OxfordPet(Dataset):
    def __init__(self, root_dir, filenames=None, transform=None, mask_transform=None):
        self.root_dir = root_dir

        if filenames:
            self.images = [os.path.join(root_dir, 'images', f"{f}.jpg") for f in filenames]
            self.masks = [os.path.join(root_dir, 'annotations', 'trimaps', f"{f}.png") for f in filenames]
        else:
            self.images = sorted(glob(os.path.join(root_dir, 'images', '*.jpg')))
            self.masks = sorted(glob(os.path.join(root_dir, 'annotations', 'trimaps', '*.png')))

        # Set transforms
        self.transform = transform if transform else get_default_transform()
        self.mask_transform = mask_transform if mask_transform else get_default_mask_transform()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        orig_w, orig_h = image.size

        # Apply transforms
        if isinstance(self.transform, A.Compose):
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = self.transform(image)
        
        # Apply mask specific transforms
        if self.mask_transform:
             if isinstance(mask, torch.Tensor):
                 if mask.ndim == 2:
                     mask = mask.unsqueeze(0)
                 mask = self.mask_transform(mask)
                 if mask.ndim == 3 and mask.shape[0] == 1:
                     mask = mask.squeeze(0)
             else:
                 mask = self.mask_transform(Image.fromarray(mask) if isinstance(mask, np.ndarray) else mask)

        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1
        
        return image, mask, (orig_w, orig_h), img_path


def split_dataset(root_dir,
                  model_type,
                  train_transform=None,
                  train_mask_transform=None,
                  val_transform=None,
                  val_mask_transform=None,
                  test_transform=None,
                  test_mask_transform=None,
                  split_dir="dataset/oxford-iiit-pet/split"):
    """
    Split the dataset according to split_dir and the model type.

    Args:
        root_dir (str): The root directory of the dataset.
        model_type (str): The type of model.
        train_transform (callable, optional)
        val_transform (callable, optional)
        test_transform (callable, optional)
        split_dir (str): The directory containing the split files.

    Returns:
        tuple(Dataset, Dataset, Dataset): A tuple containing the train, validation, and test datasets.
    """
    if model_type == "unet":
        with open(os.path.join(split_dir, 'unet', "train.txt"), "r") as f:
            train_idx = f.read().splitlines()
        with open(os.path.join(split_dir, 'unet', "val.txt"), "r") as f:
            val_idx = f.read().splitlines()
        with open(os.path.join(split_dir, 'unet', "test_unet.txt"), "r") as f:
            test_idx = f.read().splitlines()
    elif model_type == "resnet_unet":
        with open(os.path.join(split_dir, 'resnet_unet', "train.txt"), "r") as f:
            train_idx = f.read().splitlines()
        with open(os.path.join(split_dir, 'resnet_unet', "val.txt"), "r") as f:
            val_idx = f.read().splitlines()
        with open(os.path.join(split_dir, 'resnet_unet', "test_res_unet.txt"), "r") as f:
            test_idx = f.read().splitlines()
    else:
        raise ValueError("Invalid model type")
    
    train_dataset = OxfordPet(root_dir, 
                              filenames=train_idx, 
                              transform=train_transform, 
                              mask_transform=train_mask_transform)
    val_dataset = OxfordPet(root_dir, 
                            filenames=val_idx, 
                            transform=val_transform, 
                            mask_transform=val_mask_transform)
    test_dataset = OxfordPet(root_dir, 
                             filenames=test_idx, 
                             transform=test_transform, 
                             mask_transform=test_mask_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def get_default_transform(type="Torch"):
    if type == "Torch":
        return transforms.Compose([
            transforms.Resize((572, 572)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif type == "Albumentations":
        return A.Compose([
            A.Resize(height=572, width=572),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])


def get_default_mask_transform(type="Torch"):
    if type == "Torch":
        return transforms.Compose([
            transforms.Resize((572, 572), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(388),
            transforms.ToTensor(),
        ])
    elif type == "Albumentations":
        return A.Compose([
            A.CenterCrop(388),
        ])
