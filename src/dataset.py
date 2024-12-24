import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader

class AlbumentationTransforms:
    def __init__(self, mean, std):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=mean, mask_fill_value=None, p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

def get_dataloaders(mean, std, batch_size=128):
    train_transform = AlbumentationTransforms(mean, std)
    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=lambda x: test_transform(image=np.array(x))['image']
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader 