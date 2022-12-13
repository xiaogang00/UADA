import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class TinyImageNet:
    def __init__(self, batch_size, threads):
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dir = '/mnt/proj56/jqcui/Data/tiny-imagenet-200/train'
        test_dir = '/mnt/proj56/jqcui/Data/tiny-imagenet-200/val'

        train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)


    def _get_statistics(self):
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])
        train_dir = '/mnt/proj56/jqcui/Data/tiny-imagenet-200/train'
        train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

