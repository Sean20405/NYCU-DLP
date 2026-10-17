import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ICLEVRDataset(Dataset):
    def __init__(self, root_dir, json_file, objects_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            json_file (string): Path to the json file with annotations.
            objects_file (string): Path to the objects.json file for mapping.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        with open(objects_file, 'r') as f:
            self.objects_map = json.load(f)
        
        self.filenames = list(self.data.keys())
        self.num_classes = len(self.objects_map)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Multi-hot encoding for labels
        labels = self.data[img_name]
        one_hot = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.objects_map:
                one_hot[self.objects_map[label]] = 1
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "pixel_values": image,
            "labels": one_hot
        }

class ICLEVRTestDataset(Dataset):
    def __init__(self, json_file, objects_file, subset=None):
        """
        Args:
            json_file (string): Path to the test.json or new_test.json file.
            objects_file (string): Path to the objects.json file for mapping.
            subset (int, optional): Number of samples to use from the dataset.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        if subset is not None:
            self.data = self.data[:subset]

        with open(objects_file, 'r') as f:
            self.objects_map = json.load(f)
        
        self.num_classes = len(self.objects_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = self.data[idx]
        one_hot = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.objects_map:
                one_hot[self.objects_map[label]] = 1
        
        return one_hot

def get_dataloader(root_dir, json_file, objects_file, batch_size=32, shuffle=True, num_workers=12):
    dataset = ICLEVRDataset(root_dir, json_file, objects_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_test_dataloader(json_file, objects_file, batch_size=32, shuffle=False, num_workers=12, subset=None):
    dataset = ICLEVRTestDataset(json_file, objects_file, subset=subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    root_dir = "iclevr"
    train_json = "train.json"
    objects_json = "objects.json"
    
    dataset = ICLEVRDataset(root_dir, train_json, objects_json)
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample image shape: {sample['pixel_values'].shape}")
    print(f"Sample labels (multi-hot): {sample['labels']}")
    
    dataloader = get_dataloader(root_dir, train_json, objects_json, batch_size=4)
    for batch in dataloader:
        print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break
