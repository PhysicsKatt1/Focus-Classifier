##### import #####
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")


# ##### globals #####
path = r'/Users/trentstarkey/Desktop' 
train_and_val_inputs = r'/Offsets_all'
train_and_val_outputs = r'/RegressionData_30kV_0.09nA'

os.makedirs(path + train_and_val_outputs, exist_ok=True)

##### functions #####
def parse_tiff(image):
    meta_list = []
    volt_list = []
    tags = image.tag.items()
    for item in tags:
        meta_list.append(item[1][0])

    img_meta = meta_list[int(len(meta_list)-2)]
    xml = ET.fromstring(img_meta)

    amp = float(xml[3][4].text)
    volt = float(xml[3][2].text)
    hfw = float(xml[3][5][0].text)
   
    return hfw, volt, amp

##### call functions #####
labels = []
for subfolders in os.listdir(path + train_and_val_inputs):
    if '.DS_Store' in subfolders:
        continue

    for images, n in zip(os.listdir(path + train_and_val_inputs + '/' + subfolders),
                             tqdm(range(len(os.listdir(path + train_and_val_inputs + '/' + subfolders))))):
        im = Image.open(path + train_and_val_inputs + '/' + subfolders + '/' + images)
        images = images[:-4]

        #--- uncomment to save training and val data for regression model ---#
        hfw, volt, amp = parse_tiff(im)
        name = images.removeprefix('Focus_data_30000.0V_0.09nA__')
        defocus, stigx, stigy, _, _ = name.split('__')

        if float(y) == 0.0 and float(z) == 0.0:
            im.save(path + train_and_val_outputs + '/' + n + '.jpeg', format = 'JPEG')
            labels.append({'Image': n, 'Voltage': volt, 'Current': amp, 'Defocus': defocus, 'StigX': stigx, 'StigY': stigy})

all_labels = pd.DataFrame(labels)
all_labels.to_csv(path + train_and_val_outputs + '/labels.csv')
