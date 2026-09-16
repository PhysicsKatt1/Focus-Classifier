from PIL import Image
import os
from tqdm import tqdm
import cv2
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### globals #####
path = r'/Users/trentstarkey/Desktop' 
train_and_val_inputs = r'/Volumes/ThruFocusData/ThruFocusData/MixedBeams/ValData/30kV_ValData_Raw'
train_and_val_outputs_tool = r'/RegressionData_30kV_0.09nA_val_tool'
train_and_val_outputs_sim = r'/RegressionData_30kV_0.09nA_val_sim'

os.makedirs(path + train_and_val_outputs_tool, exist_ok = True)
os.makedirs(path + train_and_val_outputs_sim, exist_ok = True)

labels_tool = []
labels_sim = []
im_count = 0
sharpness = []
data_defocus = []
sigmas = np.arange(0.25, 20.0, 0.25)

##### define functions #####
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
                        data_defocus.append(abs(float(defocus)))
                        sharpness.append(cv2.Laplacian(im, cv2.CV_64F).var())

            except:
                continue


    params = np.polyfit(sharpness, data_defocus, 2)
    mod = np.poly1d(params)

    return params, mod, sharpness, data_defocus

def sort_data(params, mod):
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
                    defocus = float(defocus)

                    # save original data
                    if float(stigx) == 0.0 and float(stigy) == 0.0 and float(defocus) != 0:
                        im_count += 1
                        
                        im.save(path + train_and_val_outputs_tool + '/' + str(im_count) + '.jpeg', format = 'JPEG')
                        labels_tool.append({'Image': str(im_count), 'Voltage': 30000.0, 'Current': 0.09, 
                                       'Defocus': defocus, 'StigX': stigx, 'StigY': stigy, 'Origin': 'Tool'})

                     # create and save simulated  data
                    if float(stigx) == 0.0 and float(stigy) == 0.0 and float(defocus) == 0:
                        im = np.array(im)
                        for sigma in sigmas:
                            blurred_im = cv2.GaussianBlur(im, (0, 0), sigma)
                            blurred_im_sharpness = cv2.Laplacian(blurred_im, cv2.CV_64F).var()
                            sim_defocus = mod(blurred_im_sharpness)
    
                            im_count += 1
                            
                            Image.fromarray(blurred_im).save(path + train_and_val_outputs_sim + '/' + str(im_count) + '.jpeg', 
                                            format = 'JPEG')
                            labels_sim.append({'Image': str(im_count), 'Voltage': 30000.0, 'Current': 0.09, 
                                            'Defocus': sim_defocus, 'StigX': stigx, 'StigY': stigy, 
                                            'Origin': 'Sim', 'Sharpness':  blurred_im_sharpness})
        
            except:
                continue

    return labels_tool, labels_sim

##### call functions #####
print('Creating model fit for defocus vs sharpness')
params, mod, sharpness, data_defocus = fit_defocus()

print('Sorting true data and creating simulated data.')
labels_tool, labels_sim = sort_data(params, mod)
all_labels_tool = pd.DataFrame(labels_tool)
all_labels_tool.to_csv(path + train_and_val_outputs_tool + '/labels.csv')
all_labels_sim = pd.DataFrame(labels_sim)
all_labels_sim.to_csv(path + train_and_val_outputs_sim + '/labels.csv')

# check accuracy of linear fit and simulated data 
dummy_x = np.linspace(min(sharpness), max(sharpness), 300)

plt.scatter(sharpness, data_defocus, color = 'darkturquoise', label = 'True Data')
plt.scatter(all_labels_sim['Sharpness'], all_labels_sim['Defocus'], color='red', label = 'Sim Data')
plt.plot(dummy_x, mod(dummy_x), color = 'midnightblue', label = 'Model')
plt.title('Defocus vs Laplace Sharpness Model Fit')
plt.xlabel('Laplace Sharpness')
plt.ylabel('Defocus')
plt.legend(bbox_to_anchor = (0.9, 1.1))
plt.savefig(path + train_and_val_outputs_tool + '/model_fit.png',  bbox_inches= 'tight')