import argparse
import torch

from model.resnet import ResNet50_class200, ResNet18_class200

from model.smooth_cross_entropy import smooth_crossentropy
from data.tiny_imagenet_noaug import TinyImageNet
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR2
import sys; sys.path.append("..")
import torch.nn.functional as F

import os
from utility.cutout import Random_Crop
import numpy as np
import PIL
import random
import cv2
import numpy as np

from utility.autoaug3 import ImageNetPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.05, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--model-dir', default='./output/model-cifar-wideResNet',
                        help='directory of model for saving checkpoint')
    parser.add_argument("--epoch_resume", default=-1, type=int, help="Number of resume.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = TinyImageNet(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = ResNet50_class200().to(device)
    # model = ResNet18_class200().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    start_epoch = 0
    if args.epoch_resume > -1:
        epoch_resume = args.epoch_resume
        model_path = os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch_resume))
        optimizer_path = os.path.join(model_dir, 'opt-checkpoint_epoch{}.tar'.format(epoch_resume))
        model.load_state_dict(torch.load(model_path))
        print('load model successfully')
        optimizer.load_state_dict(torch.load(optimizer_path))
        print('load optimizer successfully')
        start_epoch = epoch_resume + 1

    scheduler = StepLR2(optimizer, args.learning_rate, args.epochs)

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
    mean = torch.Tensor(mean).float()
    std = torch.Tensor(std).float()
    crop_operation = Random_Crop(64, 64, padding=(4, 4, 4, 4))
    auto_operation = ImageNetPolicy()

    best_accuray = 0
    for epoch in range(start_epoch, args.epochs, 1):
        model.train()

        for batch in dataset.train:
            inputs_origin, targets = (b.to(device) for b in batch)

            inputs1 = []
            inputs2 = []
            inputs3 = []
            inputs4 = []
            crop_parameter = []
            flip_parameter = []
            crop_parameter_new = []
            aug_parameter = []

            aug_parameter_new = []
            for mm in range(inputs_origin.shape[0]):
                inputs_this_origin = inputs_origin[mm:mm+1].cpu()
                #### random crop
                inputs_this, i, j = crop_operation.forward(inputs_this_origin)
                inputs_this = inputs_this[0]
                #### random flip
                flip_random = random.randint(0, 1)
                inputs_this = inputs_this.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this = inputs_this.astype(np.uint8)
                if flip_random:
                    inputs_this = cv2.flip(inputs_this, 1)
                inputs_this = PIL.Image.fromarray(inputs_this)
                inputs_this, magnitude1_list, random_value_list, para_list = auto_operation.forward(inputs_this)
                inputs_this = np.array(inputs_this)
                inputs_this = torch.Tensor(inputs_this).float()
                inputs_this = inputs_this.permute(2, 0, 1)
                inputs_this = inputs_this * 1.0 / 255.0
                mean_this = mean.view(3, 1, 1).expand_as(inputs_this)
                std_this = std.view(3, 1, 1).expand_as(inputs_this)
                inputs_this = (inputs_this - mean_this) * 1.0 / std_this

                inputs1.append(inputs_this.unsqueeze(dim=0))
                crop_parameter.append([i, j])
                flip_parameter.append(flip_random)
                aug_parameter.append([magnitude1_list, random_value_list, para_list])

                #############################################################################
                ### adversarial perturbation for the data augmentation of random crop
                i_new = i + random.choice([-1, 1])
                j_new = j + random.choice([-1, 1])
                inputs_this2 = crop_operation.forward2(inputs_this_origin, i_new, j_new)
                inputs_this2 = inputs_this2[0]
                inputs_this2 = inputs_this2.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this2 = inputs_this2.astype(np.uint8)
                if (flip_random):
                    inputs_this2 = cv2.flip(inputs_this2, 1)
                inputs_this2 = PIL.Image.fromarray(inputs_this2)
                inputs_this2 = auto_operation.forward2(inputs_this2, magnitude1_list, random_value_list, para_list)
                inputs_this2 = np.array(inputs_this2)
                inputs_this2 = torch.Tensor(inputs_this2).float()
                inputs_this2 = inputs_this2.permute(2, 0, 1)
                inputs_this2 = inputs_this2 * 1.0 / 255.0
                inputs_this2 = (inputs_this2 - mean_this) * 1.0 / std_this
                inputs2.append(inputs_this2.unsqueeze(dim=0))
                crop_parameter_new.append([i_new, j_new])

                #############################################################################
                ### adversarial perturbation for the data augmentation of random flip
                i_new = i
                j_new = j
                inputs_this3 = crop_operation.forward2(inputs_this_origin, i_new, j_new)
                inputs_this3 = inputs_this3[0]
                inputs_this3 = inputs_this3.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this3 = inputs_this3.astype(np.uint8)
                if not (flip_random):
                    inputs_this3 = cv2.flip(inputs_this3, 1)
                inputs_this3 = PIL.Image.fromarray(inputs_this3)
                inputs_this3 = auto_operation.forward2(inputs_this3, magnitude1_list, random_value_list, para_list)
                inputs_this3 = np.array(inputs_this3)
                inputs_this3 = torch.Tensor(inputs_this3).float()
                inputs_this3 = inputs_this3.permute(2, 0, 1)
                inputs_this3 = inputs_this3 * 1.0 / 255.0
                inputs_this3 = (inputs_this3 - mean_this) * 1.0 / std_this
                inputs3.append(inputs_this3.unsqueeze(dim=0))

                #############################################################################
                ### adversarial perturbation for the data augmentation operations selected by RandAug
                i_new = i
                j_new = j
                inputs_this4 = crop_operation.forward2(inputs_this_origin, i_new, j_new)
                inputs_this4 = inputs_this4[0]
                inputs_this4 = inputs_this4.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this4 = inputs_this4.astype(np.uint8)
                if (flip_random):
                    inputs_this4 = cv2.flip(inputs_this4, 1)
                inputs_this4 = PIL.Image.fromarray(inputs_this4)

                magnitude1_list_new = magnitude1_list.copy()
                random_magnitude_index = random.randint(0, len(magnitude1_list_new) - 1)
                magnitude1_list_new[random_magnitude_index] = magnitude1_list_new[random_magnitude_index] + random.choice([-1, 1])

                inputs_this4 = auto_operation.forward2(inputs_this4, magnitude1_list_new, random_value_list, para_list)
                inputs_this4 = np.array(inputs_this4)
                inputs_this4 = torch.Tensor(inputs_this4).float()
                inputs_this4 = inputs_this4.permute(2, 0, 1)
                inputs_this4 = inputs_this4 * 1.0 / 255.0
                inputs_this4 = (inputs_this4 - mean_this) * 1.0 / std_this
                inputs4.append(inputs_this4.unsqueeze(dim=0))
                aug_parameter_new.append([magnitude1_list_new, random_magnitude_index])

            inputs1 = torch.cat(inputs1, dim=0).cuda()
            inputs2 = torch.cat(inputs2, dim=0).cuda()
            inputs3 = torch.cat(inputs3, dim=0).cuda()
            inputs4 = torch.cat(inputs4, dim=0).cuda()

            with torch.no_grad():
                predictions1 = model(inputs1)
                loss1 = F.cross_entropy(predictions1, targets, reduce=False)
                predictions2 = model(inputs2)
                loss2 = F.cross_entropy(predictions2, targets, reduce=False)
                predictions3 = model(inputs3)
                loss3 = F.cross_entropy(predictions3, targets, reduce=False)
                predictions4 = model(inputs4)
                loss4 = F.cross_entropy(predictions4, targets, reduce=False)
            batch_size = inputs2.shape[0]
            distance1 = loss2 - loss1
            distance2 = loss3 - loss1
            distance3 = loss4 - loss1

            inputs = []
            inputss = []
            inputsss = []
            alpha = 1.0
            for mm in range(inputs_origin.shape[0]):
                inputs_this_origin = inputs_origin[mm:mm+1].cpu()
                i, j = crop_parameter[mm]
                i_new, j_new = crop_parameter_new[mm]
                flip_random = flip_parameter[mm]
                flip_random_origin = flip_random

                magnitude1_list, random_value_list, para_list = aug_parameter[mm]
                magnitude1_list_new, random_magnitude_index = aug_parameter_new[mm]
                ########################################################################################
                ### according to the gradient, determine the value of data augmentation of random flip
                i_this = i
                j_this = j
                if distance2[mm].item() > 0:
                    flip_random = not (flip_random)
                i_this = int(i_this)
                j_this = int(j_this)
                inputs_this3 = crop_operation.forward2(inputs_this_origin, i_this, j_this)
                inputs_this3 = inputs_this3[0]
                inputs_this3 = inputs_this3.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this3 = inputs_this3.astype(np.uint8)
                if flip_random:
                    inputs_this3 = cv2.flip(inputs_this3, 1)
                inputs_this3 = PIL.Image.fromarray(inputs_this3)
                inputs_this3 = auto_operation.forward2(inputs_this3, magnitude1_list, random_value_list, para_list)
                inputs_this3 = np.array(inputs_this3)
                inputs_this3 = torch.Tensor(inputs_this3).float()
                inputs_this3 = inputs_this3.permute(2, 0, 1)
                inputs_this3 = inputs_this3 * 1.0 / 255.0
                mean_this = mean.view(3, 1, 1).expand_as(inputs_this3)
                std_this = std.view(3, 1, 1).expand_as(inputs_this3)
                inputs_this3 = (inputs_this3 - mean_this) * 1.0 / std_this
                inputs.append(inputs_this3.unsqueeze(dim=0))

                ########################################################################################
                ### according to the gradient, determine the value of data augmentation of random crop
                i_this = i + alpha * np.sign(distance1[mm].item() * 1.0 / (i_new - i))
                j_this = j + alpha * np.sign(distance1[mm].item() * 1.0 / (j_new - j))
                i_this = int(i_this)
                j_this = int(j_this)
                inputs_this4 = crop_operation.forward2(inputs_this_origin, i_this, j_this)
                inputs_this4 = inputs_this4[0]
                inputs_this4 = inputs_this4.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this4 = inputs_this4.astype(np.uint8)
                if flip_random_origin:
                    inputs_this4 = cv2.flip(inputs_this4, 1)
                inputs_this4 = PIL.Image.fromarray(inputs_this4)
                inputs_this4 = auto_operation.forward2(inputs_this4, magnitude1_list, random_value_list, para_list)
                inputs_this4 = np.array(inputs_this4)
                inputs_this4 = torch.Tensor(inputs_this4).float()
                inputs_this4 = inputs_this4.permute(2, 0, 1)
                inputs_this4 = inputs_this4 * 1.0 / 255.0
                inputs_this4 = (inputs_this4 - mean_this) * 1.0 / std_this
                inputss.append(inputs_this4.unsqueeze(dim=0))

                ########################################################################################
                ########################################################################################
                ### according to the gradient, determine the value of data augmentation operations from RandAug
                i_this = i
                j_this = j
                mag1 = magnitude1_list[random_magnitude_index]
                mag2 = magnitude1_list_new[random_magnitude_index]
                mag_this = mag1 + alpha * np.sign(distance3[mm].item() * 1.0 / (mag2 - mag1))
                mag_this = int(mag_this)
                magnitude1_list_new[random_magnitude_index] = mag_this

                i_this = int(i_this)
                j_this = int(j_this)
                inputs_this5 = crop_operation.forward2(inputs_this_origin, i_this, j_this)
                inputs_this5 = inputs_this5[0]
                inputs_this5 = inputs_this5.cpu().permute(1, 2, 0).numpy() * 255.0
                inputs_this5 = inputs_this5.astype(np.uint8)
                if flip_random_origin:
                    inputs_this5 = cv2.flip(inputs_this5, 1)
                inputs_this5 = PIL.Image.fromarray(inputs_this5)
                inputs_this5 = auto_operation.forward2(inputs_this5, magnitude1_list_new, random_value_list, para_list)
                inputs_this5 = np.array(inputs_this5)
                inputs_this5 = torch.Tensor(inputs_this5).float()
                inputs_this5 = inputs_this5.permute(2, 0, 1)
                inputs_this5 = inputs_this5 * 1.0 / 255.0
                inputs_this5 = (inputs_this5 - mean_this) * 1.0 / std_this
                inputsss.append(inputs_this5.unsqueeze(dim=0))

            inputs = torch.cat(inputs, dim=0).cuda()
            inputss = torch.cat(inputss, dim=0).cuda()
            inputsss = torch.cat(inputsss, dim=0).cuda()
            with torch.no_grad():
                predictions_adv = model(inputs)
                loss_adv = F.cross_entropy(predictions_adv, targets, reduce=False)
                predictions_adv1 = model(inputss)
                loss_adv1 = F.cross_entropy(predictions_adv1, targets, reduce=False)
                predictions_adv2 = model(inputsss)
                loss_adv2 = F.cross_entropy(predictions_adv2, targets, reduce=False)

            ### determine the data operations with the maximal loss value
            input_final = []
            for mm in range(inputs_origin.shape[0]):
                list_this = [loss_adv[mm].item(), loss_adv1[mm].item(), loss_adv2[mm].item(), loss1[mm].item()]
                max = np.argmax(list_this)
                if max == 0:
                    input_final.append(inputs[mm:mm + 1])
                elif max == 1:
                    input_final.append(inputss[mm:mm + 1])
                elif max == 2:
                    input_final.append(inputsss[mm:mm + 1])
                else:
                    input_final.append(inputs1[mm:mm + 1])
            input_final = torch.cat(input_final, dim=0)

            predictions = model(input_final)
            optimizer.zero_grad()
            loss = F.cross_entropy(predictions, targets)
            loss.backward()
            optimizer.step()
            loss = F.cross_entropy(predictions, targets, reduce=False)
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                scheduler(epoch)

        model.eval()

        total_sum = 0
        correct_sum = 0
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                total_sum += correct.shape[0]
                correct_sum += torch.sum(correct).item()
        accuracy_this = correct_sum * 1.0 / total_sum
        if accuracy_this > best_accuray:
            best_accuray = accuracy_this
            torch.save(model.state_dict(), os.path.join(model_dir, 'model-best.pt'))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch+1)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-checkpoint_epoch{}.tar'.format(epoch+1)))
