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
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, SubsetRandomSampler, dataset
from loader import *
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    filtered_batch = [item for item in batch if item[0] is not None]
    if len(filtered_batch) == 0:
        return None  # None 반환
    return default_collate(filtered_batch)

parser = argparse.ArgumentParser() # 명령줄 인자 파싱을 위한 ArgumentParser객체 생성
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
option = parser.parse_args() # 파싱된 인자 값을 option에 추가






# 현재 경로의 결과값과 모델 저장 경로 설정
os.makedirs("./truck_detection_classification/data/data/result", exist_ok=True)
os.makedirs("./truck_detection_classification/data/data/model", exist_ok=True)

loss_function = nn.CrossEntropyLoss() # 로스 펑션 크로스엔트로피로 설정

#input_shape = (option.channels, option.img_height, option.img_width) # 입력이미지의 차원 설정

resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)# 사전 훈련된 ResNet50 모델 로드

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


resnet50 = resnet50.to(device)
loss_function = loss_function.to(device)

if option.epoch != 0: # 훈련을 어느 시점(에폭)에서 재개할 것인지를 결정
    resnet50.load_state_dict("./truck_detection_classification/data/data/model%d.pth"% option.epoch)

optimizer = opt.SGD(resnet50.parameters(), lr = 0.01) # SGD 로 학습, 학습률은 0.01

def Tensor(data):
    return torch.tensor(data, device=device) # 사용할 텐서 타입을 설정합니다.

# Image transformations

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

 # need both val and train
data_dir_root = "/Users/parkjimin/Documents/24-1/캡스톤디자인/detection_project/detection/truck_detection_classification/data/data/data_altogether"
data_label_dir = "/Users/parkjimin/Documents/24-1/캡스톤디자인/detection_project/detection/truck_detection_classification/data/data/data_altogether/label"
data_pic_dir = "/Users/parkjimin/Documents/24-1/캡스톤디자인/detection_project/detection/truck_detection_classification/data/data/data_altogether/pics"

#dataset = ImageFolder(root=data_pic_dir, transform=trans)
# 데이터셋의 실제 크기를 기반으로 인덱스 생성
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
    np.random.shuffle(val_sampler.indices)

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
resnet50 = resnet50.to(device)

#sampling
#def sample_images(batches_done):
#    imgs, labels = next(iter(val_dataloader))
#    resnet50.eval()
#    images = Variable

#  Training
prev_time = time.time()
indicator = 0



train_dataset = ImageDataset(data_dir_root, data_label_dir, data_pic_dir, transform=trans)
val_dataset = ImageDataset(data_dir_root, data_label_dir, data_pic_dir, transform=trans)

# 데이터 로더 설정
train_dataloader = DataLoader(
    train_dataset,
    batch_size=option.batch_size,
    shuffle=False,
    num_workers=option.n_cpu,
    sampler=train_sampler,
    collate_fn=custom_collate_fn  # 커스텀 collate 함수 사용
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=5,  # 검증 데이터 로더의 배치 크기 설정
    shuffle=False,
    num_workers=0,
    sampler=val_sampler,
    collate_fn=custom_collate_fn  # 커스텀 collate 함수 사용
)

for epoch in range(option.n_epochs):
    running_loss = 0.0

    for i, batch in enumerate(train_dataloader):
        if batch is None:  # 배치가 None인 경우 건너뛰기
            print("Skipping empty batch.")
            continue

        inputs, labels = batch
        inputs = inputs.to(device)  # 입력 데이터를 디바이스로 이동

        # 빈 배치 검사
        if inputs.nelement() == 0:
            print("Skipping empty batch.")
            continue

        # 레이블을 디바이스로 이동
        labels = torch.tensor([classes_mapping[label] if label in classes_mapping else -1 for label in labels],
                              dtype=torch.long).to(device)

        optimizer.zero_grad()

        # 모델에 입력
        outputs = resnet50(inputs)
        outputs = F.softmax(outputs, dim=1)

        # 레이블 변환 중 잘못된 레이블을 처리
        if (labels == -1).any():
            print("Invalid labels found, skipping batch")
            continue

        # 손실 계산 및 역전파
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:  # 매 2000번째 배치마다 로그 출력
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


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
        if batch is None:  # 배치가 None인 경우 건너뛰기
            print("Skipping empty batch.")
            continue
        inputs, labels = batch

        inputs = inputs.to(device)

        outputs = resnet50(inputs)
        outputs = F.softmax(outputs, dim=1)


        labels = torch.tensor([classes_mapping[label] if label in classes_mapping else -1 for label in labels],
                              dtype=torch.long).to(device)
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