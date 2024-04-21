import glob
import random
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
import json

class ImageDataset(Dataset):
    def __init__(self, annotations_file,label_dir, img_dir, transform=None, target_transform=None):
        df = pd.DataFrame()
        for filename in os.listdir(os.path.join(annotations_file,label_dir)):
            if os.path.isfile(os.path.join(annotations_file,label_dir, filename)):
                with open(os.path.join(annotations_file,label_dir, filename), 'rt', encoding='UTF8') as file:
                    base_name, extension = os.path.splitext(filename)
                    image_filename = base_name + '.jpg'
                    if os.path.exists(os.path.join(annotations_file, img_dir, image_filename)):
                        data = json.load(file)
                        key_data = data['FILE'][0]['ITEMS'][0]['SEGMENT']

                        new_set = {"filename": [image_filename], "SEGMENT": [key_data]}
                        df_temp = pd.DataFrame(new_set)

                        df = pd.concat([df, df_temp])
                        df.reset_index(drop=True, inplace=True)

        self.image_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        print(len(df))
        print('init done')

    def __getitem__(self, idx):
        ## 여기서 결정해야되는것 label에 있는 데이터만 써서 할거냐 아니면 crop하다 나온 겹친 이미지도 이름비교해서 쓸수있게 할거냐
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        ## 우선은 label에 있는것만 하는걸로
        return len(self.image_labels)
