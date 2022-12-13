from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random



class CIFARPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR.
        Example:
        ## >>> policy = CIFARPolicy()
        # >>> transformed = policy(image)
        Example as a PyTorch Transform:
        # >>> transform=transforms.Compose([
        # >>>     transforms.Resize(256),
        # >>>     CIFARPolicy(),
        # >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),  #
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),  #
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "identity": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random_value, 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random_value, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random_value, 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random_value),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude, random_value: rotate_with_fill(img, magnitude),  ###
            "color": lambda img, magnitude, random_value: ImageEnhance.Color(img).enhance(1 + magnitude * random_value),
            "posterize": lambda img, magnitude, random_value: ImageOps.posterize(img, magnitude),  ###
            "solarize": lambda img, magnitude, random_value: ImageOps.solarize(img, magnitude),  ###
            "contrast": lambda img, magnitude, random_value: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random_value),
            "sharpness": lambda img, magnitude, random_value: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random_value),
            "brightness": lambda img, magnitude, random_value: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random_value),
            "autocontrast": lambda img, magnitude, random_value: ImageOps.autocontrast(img),  ###
            "equalize": lambda img, magnitude, random_value: ImageOps.equalize(img),  ###
            "identity": lambda img, magnitude, random_value: img  ###
            ## "invert": lambda img, magnitude, random_value: ImageOps.invert(img)  ###
        }
        self.func = func
        self.ranges = ranges

    def forward(self, img):
        num_choose = 2
        parameter_list = ["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize",
                          "contrast", "sharpness", "brightness", "autocontrast", "equalize", "identity"]
        parameter_list_choose = []
        for mm in range(num_choose):
            random_index = random.randint(0, len(parameter_list)-1)
            parameter_list_choose.append(parameter_list[random_index])

        magnitude1_list = []
        random_value_list = []
        for mm in range(len(parameter_list_choose)):
            operation1 = self.func[parameter_list_choose[mm]]
            magnitude_idx1 = random.randint(0, 9)
            magnitude1 = self.ranges[parameter_list_choose[mm]][magnitude_idx1]
            random_value1 = random.choice([-1, 1])
            img = operation1(img, magnitude1, random_value1)
            magnitude1_list.append(magnitude_idx1)
            random_value_list.append(random_value1)

        return img, magnitude1_list, random_value_list, parameter_list_choose

    def forward2(self, img, magnitude1_list, random_value_list, parameter_list_choose):
        for mm in range(len(parameter_list_choose)):
            operation1 = self.func[parameter_list_choose[mm]]
            magnitude_idx1 = magnitude1_list[mm]
            magnitude_idx1 = max(magnitude_idx1, 0)
            magnitude_idx1 = min(magnitude_idx1, 9)
            magnitude1 = self.ranges[parameter_list_choose[mm]][magnitude_idx1]
            random_value1 = random_value_list[mm]
            img = operation1(img, magnitude1, random_value1)
        return img

    def __repr__(self):
        return "AutoAugment CIFAR Policy"



class ImageNetPolicy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),  #
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),  #
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "identity": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random_value, 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random_value, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random_value, 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude, random_value: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random_value),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude, random_value: rotate_with_fill(img, magnitude),  ###
            "color": lambda img, magnitude, random_value: ImageEnhance.Color(img).enhance(1 + magnitude * random_value),
            "posterize": lambda img, magnitude, random_value: ImageOps.posterize(img, magnitude),  ###
            "solarize": lambda img, magnitude, random_value: ImageOps.solarize(img, magnitude),  ###
            "contrast": lambda img, magnitude, random_value: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random_value),
            "sharpness": lambda img, magnitude, random_value: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random_value),
            "brightness": lambda img, magnitude, random_value: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random_value),
            "autocontrast": lambda img, magnitude, random_value: ImageOps.autocontrast(img),  ###
            "equalize": lambda img, magnitude, random_value: ImageOps.equalize(img),  ###
            "identity": lambda img, magnitude, random_value: img  ###
            ## "invert": lambda img, magnitude, random_value: ImageOps.invert(img)  ###
        }
        self.func = func
        self.ranges = ranges

    def forward(self, img):
        num_choose = 2
        parameter_list = ["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize",
                          "contrast", "sharpness", "brightness", "autocontrast", "equalize", "identity"]
        parameter_list_choose = []
        for mm in range(num_choose):
            random_index = random.randint(0, len(parameter_list)-1)
            parameter_list_choose.append(parameter_list[random_index])

        magnitude1_list = []
        random_value_list = []
        for mm in range(len(parameter_list_choose)):
            operation1 = self.func[parameter_list_choose[mm]]
            magnitude_idx1 = random.randint(0, 9)
            magnitude1 = self.ranges[parameter_list_choose[mm]][magnitude_idx1]
            random_value1 = random.choice([-1, 1])
            img = operation1(img, magnitude1, random_value1)
            magnitude1_list.append(magnitude_idx1)
            random_value_list.append(random_value1)

        return img, magnitude1_list, random_value_list, parameter_list_choose

    def forward2(self, img, magnitude1_list, random_value_list, parameter_list_choose):
        for mm in range(len(parameter_list_choose)):
            operation1 = self.func[parameter_list_choose[mm]]
            magnitude_idx1 = magnitude1_list[mm]
            magnitude_idx1 = max(magnitude_idx1, 0)
            magnitude_idx1 = min(magnitude_idx1, 9)
            magnitude1 = self.ranges[parameter_list_choose[mm]][magnitude_idx1]
            random_value1 = random_value_list[mm]
            img = operation1(img, magnitude1, random_value1)
        return img

    def __repr__(self):
        return "AutoAugment ImageNet Policy"
