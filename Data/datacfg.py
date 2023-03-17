from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch

class UlcerData(Dataset):
    def __init__(self, imagePath, maskPath, transforms, transform2):
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.transforms = transforms
        self.transform2 = transform2
        self.all_images = os.listdir(imagePath)
        self.all_labels = os.listdir(maskPath)
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.imagePath, self.all_images[idx])
        label_loc = os.path.join(self.maskPath, self.all_labels[idx])
        image = Image.open(img_loc)
        label = Image.open(label_loc).convert('L')
        tsr = self.transforms(image)
        return (tsr,self.transform2(label))