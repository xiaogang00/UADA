import argparse
import torch

from model.resnet import ResNet50_class200, ResNet18_class200

from model.smooth_cross_entropy import smooth_crossentropy
from data.tiny_imagenet_noaug import TinyImageNet
from utility.initialize import initialize
import sys; sys.path.append("..")

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--model_dir', default='./output/model-cifar-wideResNet',
                        help='directory of model for saving checkpoint')
    parser.add_argument("--epoch_resume", default=-1, type=int, help="Number of resume.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = TinyImageNet(args.batch_size, args.threads)
    # model = ResNet18_class200().to(device)
    model = ResNet50_class200().to(device)

    model_dir = args.model_dir
    model_path = os.path.join(model_dir, 'model-best.pt')
    model.load_state_dict(torch.load(model_path))
    print('load model successfully')

    model.eval()

    total_sum = 0
    correct_sum = 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            print(torch.sum(correct), correct.shape[0])
            total_sum += correct.shape[0]
            correct_sum += torch.sum(correct).item()
    print(correct_sum * 1.0 / total_sum)
