import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset,random_split

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




class MNISTDataset(Dataset):
    def __init__(self,path,transform=None):
        self.path=path
        self.transform=transform

        self.len_dataset = 0
        self.data_list = []

        for part_dir,dir_list,file_list in os.walk(path):
            if part_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_indx = {
                    cls_name:i for i,cls_name in enumerate(self.classes)
                }
                continue
            cls = part_dir.split("/")[-1]

            # формируем путь до конкретной картинке .jpg
            for name_file in file_list:
                file_path = os.path.join(part_dir,name_file)
                self.data_list.append((file_path,self.class_to_indx[cls]))
            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset
    def __getitem__(self, item):
        file_path,target = self.data_list[item]
        sample = np.array(Image.open(file_path))

        if self.transform:
            sample = self.transform(sample)

        return sample, target

train_data = MNISTDataset(r"/content/mnist/training")
test_data = MNISTDataset(r"/content/mnist/testing")

#апроверяем классы
train_data.classes

train_data.class_to_indx

for cls, one_hot_position in train_data.class_to_indx.items():
    one_hot_vector = [(i == one_hot_position)*1 for i in range(10)]
    print(f"\033[32m{cls}\033[0m => \033[34m{one_hot_vector}\033[0m")

#len
print(f"Длина трен данных: {len(train_data)}")
print(f"Длина тестовых данных: {len(test_data)}")

#getitem
train_data[2564]

img, one_hot_position = train_data[59999]
cls = train_data.classes[one_hot_position]
print(f"Класс - {cls}")
plt.imshow(img, cmap ="gray")

train_data, val_data = random_split(train_data,[0.8,0.2])

print(f"Длина трен данных: {len(train_data)}")
print(f"Длина валид данных: {len(val_data)}")
print(f"Длина тестовых данных: {len(test_data)}")

train_loader = DataLoader(train_data,batch_size=16,shuffle=True)
val_loader = DataLoader(val_data,batch_size=16,shuffle=False)
test_loader = DataLoader(test_data,batch_size=16,shuffle=False)

for i ,(samples, target) in enumerate(train_loader):
    if i < 3:
        print(f"Номер batch = {i+1}")
        print(f"   размер samples = {samples.shape}")
        print(f"   размер target = {target.shape}")

print("\n    ..............   \n")
print(f"Номер batch = {i+1}")
print(f"   размер samples = {samples.shape}")
print(f"   размер target = {target.shape}")