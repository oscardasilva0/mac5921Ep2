import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class DriveDataset(Dataset):
    def __init__(self, root_path, test=False, resize_size=(512, 512)):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
        else:
            # print(os.listdir(root_path + "/train/"))
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)


import albumentations as A

# Define um pipeline de aumento de dados
train_transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomCrop(width=128, height=128),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Adiciona Gaussian Blur com probabilidade de 50%
    # ... Adicione mais transformações conforme necessário ...
])

# Aplique a transformação ao carregar as imagens no dataset
class DriveDatasetWithAugmentation(DriveDataset):
    def __init__(self, data_path, resize_size, transform=None):
        super().__init__(data_path, resize_size)
        self.transform = transform

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)

        # Aplique o aumento de dados
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask
