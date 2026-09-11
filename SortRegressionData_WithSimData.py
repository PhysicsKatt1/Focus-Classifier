from PIL import Image
import os
from tqdm import tqdm
import cv2
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

# ##### globals #####
path = r'/Users/trentstarkey/Desktop' 
train_and_val_inputs = r'/Volumes/ThruFocusData/ThruFocusData/MixedBeams/ValData/30kV_ValData_Raw'
train_and_val_outputs = r'/RegressionData_30kV_0.09nA_val'

os.makedirs(path + train_and_val_outputs, exist_ok = True)

labels = []
im_count = 0
sharpness = []
data_defocus = []
sigmas = np.arange(0.25, 20.0, 0.25)

def fit_defocus():
    for subfolders in os.listdir(train_and_val_inputs):
        if '.DS_Store' in subfolders:
            continue

        for images, n in zip(os.listdir(train_and_val_inputs + '/' + subfolders),
                                tqdm(range(len(os.listdir(train_and_val_inputs + '/' + subfolders))))):

            try:
                im = Image.open(train_and_val_inputs + '/' + subfolders + '/' + images).convert('L')
                im = np.array(im)

                if 'Focus_data_30000.0V_0.09nA__' in images:
                    name = images.removeprefix('Focus_data_30000.0V_0.09nA__')
                    defocus, stigx, stigy, _, _ = name.split('__')

                    if float(stigx) == 0.0 and float(stigy) == 0.0 and float(defocus) != 0:
                        data_defocus.append(defocus)
                        sharpness.append(cv2.Laplacian(im, cv2.CV_64F).var())

            except:
                continue

    def linear_mod(x, a, b):
        return a * x + b

    params, _ = curve_fit(linear_mod, sharpness, data_defocus)

    return params, linear_mod

def sort_data(params, linear_mod):
    global im_count 

    for subfolders in os.listdir(train_and_val_inputs):
        if '.DS_Store' in subfolders:
            continue

        for images, n in zip(os.listdir(train_and_val_inputs + '/' + subfolders),
                                tqdm(range(len(os.listdir(train_and_val_inputs + '/' + subfolders))))):

            try:
                im = Image.open(train_and_val_inputs + '/' + subfolders + '/' + images).convert('L')
                images = images[:-4]

                if 'Focus_data_30000.0V_0.09nA__' in images:
                    name = images.removeprefix('Focus_data_30000.0V_0.09nA__')
                    defocus, stigx, stigy, _, _ = name.split('__')

                    if float(stigx) == 0.0 and float(stigy) == 0.0 and float(defocus) != 0:
                        im_count += 1
                        
                        im.save(path + train_and_val_outputs + '/' + str(im_count) + '.jpeg', format = 'JPEG')
                        labels.append({'Image': str(im_count), 'Voltage': 30000.0, 'Current': 0.09, 
                                       'Defocus': defocus, 'StigX': stigx, 'StigY': stigy, 'Origin': 'Tool'})

                    elif float(stigx) == 0.0 and float(stigy) == 0.0 and float(defocus) == 0:
                        for sigma in sigmas:
                            im = np.array(im)
                            blurred_im = cv2.GaussianBlur(im, (0, 0), sigma)
                            blurred_im_sharpness = cv2.Laplacian(blurred_im, cv2.CV_64F).var()
                            sim_defocus = linear_mod(blurred_im_sharpness, *params)
    
                            im_count += 1
                            
                            Image.fromarray(blurred_im).save(path + train_and_val_outputs + '/' + str(im_count) + '.jpeg', 
                                            format = 'JPEG')
                            labels.append({'Image': str(im_count), 'Voltage': 30000.0, 'Current': 0.09, 
                                            'Defocus': sim_defocus, 'StigX': stigx, 'StigY': stigy, 
                                            'Origin': 'Sim'})
        
            except:
                continue

    return labels


params, linear_mod = fit_defocus()
labels = sort_data(params, linear_mod)

all_labels = pd.DataFrame(labels)
all_labels.to_csv(path + train_and_val_outputs + '/labels.csv')