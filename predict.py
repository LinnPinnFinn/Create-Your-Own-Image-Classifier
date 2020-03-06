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
parser = argparse.ArgumentParser(description='Predicting')

parser.add_argument('--image_path', dest='image_path', action='store', default='flowers/test/73/image_00291.jpg')
parser.add_argument('--data_dir', dest='data_dir', action='store', default='flowers')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
parser.add_argument('--topk', type=int, dest='topk', action='store', default='5')
parser.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json')
parser.add_argument('--device', dest='device', action='store', default='cuda', choices=['cuda', 'cpu'])
    
pa = parser.parse_args()

def main():
    train_data, validation_data, test_data = utils.data_transforms(pa.data_dir)
    trainloader, validationloader, testloader = utils.data_loaders(pa.data_dir)
    
    model = utils.load_checkpoint(pa.save_dir)
    
    with open(pa.category_names) as json_file:
        cat_to_name = json.load(json_file)
        
    probs, classes = utils.predict(pa.image_path, model, pa.topk, pa.device)
    
    probs = probs.type(torch.FloatTensor).to('cpu').numpy()
    classes = classes.type(torch.FloatTensor).to('cpu').numpy()
    classes = classes.astype(int)
    classes = classes.astype(str)

    class_names = [cat_to_name[i] for i in classes[0]]

    print(probs)
    print(classes)
    print(class_names)
    
    print("Finsihed predicting!")
    
main()