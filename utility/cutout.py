import torch
import torch.nn.functional as F
import random


class Cutout2():
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def forward(self, image):
        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image, left, top, right, bottom

    def forward2(self, image, left, top, right, bottom):
        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


class Random_Crop():
    def __init__(self, tw, th, padding, pad_if_needed=False, fill=0, padding_mode="constant"):
        self.tw = tw
        self.th = th
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.size = [th, tw]

    def forward(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.padding_mode, self.fill)
        h = img.shape[2]
        w = img.shape[3]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)
        return img[:, :, i:i+self.th, j:j+self.tw], i, j

    def forward2(self, img, i, j):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.padding_mode, self.fill)
        i = max(0, i)
        j = max(0, j)
        h = img.shape[2]
        w = img.shape[3]
        i = min(h - self.th, i)
        j = min(w - self.tw, j)
        return img[:, :, i:i+self.th, j:j+self.tw]

