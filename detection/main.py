import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch
import torchvision.models as models
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, SubsetRandomSampler
from loader import *
from torchvision import datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=224, help="size of image height")
parser.add_argument("--img_width", type=int, default=224, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
option = parser.parse_args()

cuda = torch.cuda.is_available()



os.makedirs(r"C:\truck_detection_classification\data\data\result", exist_ok=True)
os.makedirs(r"C:\truck_detection_classification\data\data\model", exist_ok=True)

loss_function = nn.CrossEntropyLoss()

input_shape = (option.channels, option.img_height, option.img_width)

resnet50 = models.resnet50(input_shape, pretrained=True)

if cuda:
    resnet50 = resnet50.cuda()
    loss_function = loss_function.cuda()

if option.epoch != 0:
    resnet50.load_state_dict("C:\truck_detection_classification\data\data\model%d.pth"% option.epoch)

optimizer = opt.SGD(resnet50.parameters(), lr = 0.01)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

 # need both val and train
data_dir_root = r"C:\truck_detection_classification\data\data\data_altogether"
data_label_dir = r"C:\truck_detection_classification\data\data\data_altogether\label"
data_pic_dir = r"C:\truck_detection_classification\data\data\data_altogether\pics"

data_size = 15866
indices = list(range(data_size))

split_ratio = 0.8
split = int(split_ratio * data_size)
train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

if isinstance(train_sampler, SubsetRandomSampler):
    np.random.shuffle(train_sampler.indices)

if isinstance(val_sampler, SubsetRandomSampler):
    np.random.shuffle(train_sampler.indices)

print("Length of train_indices:", len(train_indices))
print("Length of val_indices:", len(val_indices))

train_dataloader = DataLoader(ImageDataset(data_dir_root,data_label_dir, data_pic_dir, transform=trans),
                              batch_size=option.batch_size,shuffle=False,num_workers=option.n_cpu,sampler=train_sampler)

val_dataloader = DataLoader(ImageDataset(data_dir_root,data_label_dir, data_pic_dir, transform=trans),
                            batch_size=5, shuffle=False,num_workers=0,sampler=val_sampler)
num_classes = 3
classes = ('대형차', '중형차','소형차')
classes_mapping = {'대형차':0, '중형차':1, '소형차':2}

resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

#sampling
#def sample_images(batches_done):
#    imgs, labels = next(iter(val_dataloader))
#    resnet50.eval()
#    images = Variable

#  Training
prev_time = time.time()
indicator = 0
for epoch in range(option.n_epochs):
    running_loss = 0.0

    for i, batch in enumerate(train_dataloader):
        inputs, labels = batch
        optimizer.zero_grad()

        outputs = resnet50(inputs)
        outputs = F.softmax(outputs, dim=1)
        labels = torch.tensor([classes_mapping[label] for label in labels])
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        indicator = indicator+1
print(indicator)
print('Finished Training')

# Validation loop
resnet50.eval()  # Set model to evaluation mode
val_loss = 0.0
correct = 0
total = 0

indicator = 0
with torch.no_grad():  # Disable gradient calculation during validation
    for batch in val_dataloader:
        inputs, labels = batch

        outputs = resnet50(inputs)
        outputs = F.softmax(outputs, dim=1)


        labels = torch.tensor([classes_mapping[label] for label in labels])
        #print('out-lab %d', i)
        #print(outputs, labels)

        loss = loss_function(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        #print('pre-lab %d',i)
        #print( predicted, labels)
        print((predicted == labels).sum().item())
        correct += (predicted == labels).sum().item()
        indicator=indicator+1

print(correct)
print(total)
print(indicator)
# Calculate validation loss and accuracy
avg_val_loss = val_loss / len(val_dataloader)
val_accuracy = correct / total

print(f'Validation loss: {avg_val_loss:.3f}, Accuracy: {val_accuracy:.3f}')

#total = 0
#correct = 0
#with torch.no_grad():
#    for i,batch in enumerate(val_dataloader):
#        inputs, labels = batch
#        outputs = resnet50(inputs)

#        _, predictions = torch.max(outputs, 1)
#        print(labels)
#        print(predictions)
#        total += labels.size(0)
#        correct += (predictions == labels).sum().item()

#print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#correct_pred = {classname: 0 for classname in classes}
#total_pred = {classname: 0 for classname in classes}