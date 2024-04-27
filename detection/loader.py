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
        print("Annotations file path:", annotations_file)
        print("Label directory path:", label_dir)
        combined_path = os.path.join(annotations_file, label_dir)
        print("Combined path:", combined_path)
        print("Image_dir path:", img_dir)

        for filename in os.listdir(os.path.join(annotations_file,label_dir)): # 디렉토리 모든 파일
            if os.path.isfile(os.path.join(annotations_file,label_dir, filename)): # 실제 파일명인지 체크
                with open(os.path.join(annotations_file,label_dir, filename), 'rt', encoding='UTF8') as file: # JSON파일을 UTF-8인코딩으로 확인
                    base_name, extension = os.path.splitext(filename) # 파일 이름에서 확장자 분리
                    image_filename = base_name + '.jpg' #JSON파일과 동일한 파일 이름 생성
                    if os.path.exists(os.path.join(annotations_file, img_dir, image_filename)): #해당 이미지 파일 존재 체크
                        data = json.load(file) #JSON 파일 내용 파싱
                        key_data = data['FILE'][0]['ITEMS'][0]['SEGMENT'] #JSON 구조에서 필요한 데이터를 추출합니다.

                        new_set = {"filename": [image_filename], "SEGMENT": [key_data]} #추출된 데이터를 새로운 딕셔너리로 구성합니다.
                        df_temp = pd.DataFrame(new_set) # 임시 데이터프레임을 생성하여 파일 이름과 레이블 데이터를 저장합니다.

                        df = pd.concat([df, df_temp]) # 임시 데이터프레임을 메인 데이터프레임에 연결합니다.
                        df.reset_index(drop=True, inplace=True) # 데이터프레임의 인덱스를 재설정합니다.

        self.image_labels = df # 이미지 파일 이름과 레이블을 저장하는 데이터프레임
        self.img_dir = img_dir # 이미지 파일이 저장된 디렉터리 경로
        self.transform = transform # 이미지 변환 함수를 클래스 속성으로 설정합니다.
        self.target_transform = target_transform # 레이블 변환 함수
        print(len(df))
        print('init done')

    def __getitem__(self, idx): #인덱스에 해당하는 이미지와 레이블을 반환하는 메소드
        ## 여기서 결정해야되는것 label에 있는 데이터만 써서 할거냐 아니면 crop하다 나온 겹친 이미지도 이름비교해서 쓸수있게 할거냐
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0]) # 데이터프레임에서 인덱스에 해당하는 이미지 파일의 경로를 구성
        # 파일 존재 여부를 확인
        if os.path.exists(img_path):
            print("File exists, trying to open.")
            image = Image.open(img_path)  # PIL 라이브러리를 사용하여 이미지 파일을 엽니다
        else:
            print("File not found:", img_path)
            return None, None  # 파일이 없을 경우 None 반환 # PIL 라이브러리를 사용하여 이미지 파일을 엽니다
        label = self.image_labels.iloc[idx, 1] #데이터프레임에서 인덱스에 해당하는 레이블 데이터를 가져옵니다.
        if self.transform: #이미지 변환 함수가 설정되어 있다면, 해당 함수를 이미지에 적용합니다.
            image = self.transform(image)
        if self.target_transform: # 레이블 변환 함수가 설정되어 있다면, 해당 함수를 레이블에 적용합니다.
            label = self.target_transform(label)
        return image, label # 변환된 이미지와 라벨 반환

    def __len__(self):
        ## 우선은 label에 있는것만 하는걸로
        return len(self.image_labels)
