# Imports
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import time
from PIL import Image
import numpy as np
import argparse
import utils

# Command line parser object and arguments 
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir', dest='data_dir', action='store', default='flowers')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
parser.add_argument('--architecture', dest='architecture', action='store', default='resnet101', choices=['resnet101', 'densenet161', 'vgg16'])
parser.add_argument('--dropout', type=float, dest='dropout', action='store', default='0.2')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', action='store', default='0.001')
parser.add_argument('--input_units', type=int, dest='input_units', action='store', default='2048')
parser.add_argument('--hidden_units', type=int, dest='hidden_units', action='store', default='512')
parser.add_argument('--epochs', type=int, dest='epochs', action='store', default='18')
parser.add_argument('--print_every', type=int, dest='print_every', action='store', default='15')
parser.add_argument('--device', dest='device', action='store', default='cuda', choices=['cuda', 'cpu'])
    
pa = parser.parse_args()

def main():
    train_data, validation_data, test_data = utils.data_transforms(pa.data_dir)
    trainloader, validationloader, testloader = utils.data_loaders(pa.data_dir)
    
    model, criterion, optimizer = utils.network_setup(pa.architecture, pa.dropout, pa.input_units, pa.hidden_units,         pa.learning_rate, pa.device)
    
    utils.network_training(model, trainloader, validationloader, criterion, optimizer, pa.epochs, pa.print_every, pa.device)
    
    utils.save_checkpoint(model, train_data, optimizer, pa.architecture, pa.dropout, pa.input_units, pa.hidden_units,   pa.learning_rate, pa.epochs, pa.save_dir)
    
    print("Finished training!")

main()