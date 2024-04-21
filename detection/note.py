import glob
import random
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
import json




#for filename in os.listdir(f""):
 #   if os.path.isfile(os.path.join(annotations_file, filename)):
  #      with open(filename) as file:
   #         data = json.load(file)
    #        key_data = data['SEGMENT']
     #       df_temp = pd.DataFrame(key_data)
      #      merged_df = pd.merge(df, df_temp, on='SEGMENT', how='outer')

annotations_file = r"C:\truck_detection_classification\data\data\data_altogether\label"

file_r =  r"A01_B02_C00_D01_0701_E05_F06_202_1.json"
file_t = r"A01_B02_C00_D01_0701_E05_F06_202_2.json"


df = pd.DataFrame()
for filename in os.listdir(annotations_file):
    if os.path.isfile(os.path.join(annotations_file, filename)):
        #print(os.path.join(annotations_file, filename))
        with open(os.path.join(annotations_file, filename),'rt', encoding='UTF8') as file:

            data = json.load(file)
            key_data = data['FILE'][0]['ITEMS'][0]['SEGMENT']

            new_set = {"filename": [filename], "SEGMENT": [key_data]}
            df_temp = pd.DataFrame(new_set)

            df = pd.concat([df, df_temp])
            df.reset_index(drop=True, inplace=True)

print(df.to_string(index=True))

#with open(filename+file_t, 'rt', encoding='UTF8') as file:
 #   data = json.load(file)
  #  key_data = data['FILE'][0]['ITEMS'][0]['SEGMENT']
   # print("seperate\n")
#
 #   new_set = {"filename" :[file_t], "SEGMENT":[key_data]}
  #  df_temp1 = pd.DataFrame(new_set)
#
#df = pd.concat([df, df_temp])
#df.reset_index(drop=True, inplace=True)
#print(df.to_string(index=True))


#temp = []
#        for i in range(num_classes):
 #           if (classes_mapping[label] for label in labels) == i:
  #              temp.append(1)
   #         else:
    #            temp.append(0)
     #   labels = torch.tensor(temp)