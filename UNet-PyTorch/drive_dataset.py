import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class DriveDataset(Dataset):
    def __init__(self, root_path, test=False, resize_size=(512, 512), transform=None):
        self.root_path = root_path
        self.resize_size = resize_size
        self.transform = transform

        if test:
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
        else:
            print(os.listdir(root_path + "/train/"))
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])


    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Aplica o resize ANTES da transformação
        img = transforms.Resize(self.resize_size)(img)
        mask = transforms.Resize(self.resize_size)(mask)

        # Aplica as transformações personalizadas, se fornecidas
        if self.transform is not None:
            img = self.transform(img)

        # Aplica transforms.ToTensor() se img ainda não for um tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        mask = transforms.ToTensor()(mask)

        return img, mask

    def __len__(self):
        return len(self.images)


# import albumentations as A

# # Define um pipeline de aumento de dados
# train_transform = A.Compose([
#     A.RandomRotate90(),
#     A.HorizontalFlip(),
#     A.VerticalFlip(),
#     A.RandomCrop(width=128, height=128),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
#     A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Adiciona Gaussian Blur com probabilidade de 50%
#     # ... Adicione mais transformações conforme necessário ...
# ])

# # Aplique a transformação ao carregar as imagens no dataset
# class DriveDatasetWithAugmentation(DriveDataset):
#     def __init__(self, data_path, resize_size, transform=None):
#         super().__init__(data_path, resize_size)
#         self.transform = transform

#     def __getitem__(self, idx):
#         img, mask = super().__getitem__(idx)

#         # Aplique o aumento de dados
#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']

#         return img, mask
